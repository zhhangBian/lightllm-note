import re

from typing import List, Tuple
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    CLIPVisionModel,
    CLIPVisionConfig,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from .configuration_mineru2 import Mineru2QwenConfig
from .image_processing_mineru2 import Mineru2ImageProcessor, expand2square, process_anyres_image

from lightllm.server.multimodal_params import ImageItem
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.utils.log_utils import init_logger

import os
from safetensors import safe_open

logger = init_logger(__name__)


def build_vision_tower(config: Mineru2QwenConfig):
    vision_tower = getattr(config, "mm_vision_tower", getattr(config, "vision_tower", ""))
    model_path = getattr(config, "_name_or_path", "")

    def _resolve_path(name):
        if model_path:
            return f"{model_path}/{name}"
        return name

    if "clip" in vision_tower.lower():
        vt_path = _resolve_path(vision_tower)
        print(f"[debug] load clip from {vt_path}")
        return CLIPVisionModel.from_pretrained(vt_path)
    elif "siglip" in vision_tower.lower():
        vt_path = _resolve_path(vision_tower)
        print(f"[debug] load siglip from {vt_path}")
        model = SiglipVisionModel.from_pretrained(vt_path)
        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = max(0, model.config.num_hidden_layers - 1)
        if hasattr(model, "config") and hasattr(model.config, "vision_use_head"):
            model.config.vision_use_head = False
        return model
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")


def build_vision_projector(config: Mineru2QwenConfig):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return nn.Identity()

    raise ValueError(f"Unknown projector type: {projector_type}")


class Mineru2VisionModel:
    def __init__(self):
        pass

    def _load_projector_weights(self, weight_dir: str):
        # 扫描 safetensors/bin 文件并尝试加载 projector 权重
        def iter_state_dicts(dir_path: str):
            for f in os.listdir(dir_path):
                full = os.path.join(dir_path, f)
                if not os.path.isfile(full):
                    continue
                if f.endswith(".safetensors"):
                    try:
                        with safe_open(full, framework="pt", device="cpu") as sf:
                            yield {k: sf.get_tensor(k) for k in sf.keys()}
                    except Exception as e:
                        print(f"[warning] safetensors read fail: {full} due to {e}")
                elif f.endswith(".bin"):
                    try:
                        state = torch.load(full, map_location="cpu")
                        if isinstance(state, dict):
                            yield state
                    except Exception as e:
                        print(f"[warning] bin read fail: {full} due to {e}")

        def assign_linear(linear: nn.Linear, w: torch.Tensor = None, b: torch.Tensor = None):
            if w is not None:
                linear.weight.data.copy_(w.to(dtype=linear.weight.dtype))
            if b is not None and linear.bias is not None:
                linear.bias.data.copy_(b.to(dtype=linear.bias.dtype))

        def try_assign_from_keydict(key_to_tensor: dict) -> bool:
            # 兼容命名：
            # - 线性：model.mm_projector.(weight|bias) / model.mm_projector.linear.(weight|bias)
            # - 2层MLP：model.mm_projector.{0,2}.(weight|bias)
            # - LLaVA风格别名：multi_modal_projector.linear_1 / linear_2
            if len(linear_modules) == 1:
                w = None
                b = None
                for k in ("model.mm_projector.weight", "model.mm_projector.linear.weight"):
                    if k in key_to_tensor:
                        w = key_to_tensor[k]
                        break
                for k in ("model.mm_projector.bias", "model.mm_projector.linear.bias"):
                    if k in key_to_tensor:
                        b = key_to_tensor[k]
                        break
                if w is not None:
                    assign_linear(linear_modules[0], w, b)
                    print("[debug] projector load: single Linear matched")
                    return True
                # 兜底：若权重以分层形式存在，且本地只有一层，则尝试用第一层
                for k in ("model.mm_projector.0.weight", "multi_modal_projector.linear_1.weight"):
                    if k in key_to_tensor:
                        w = key_to_tensor[k]
                        break
                for k in ("model.mm_projector.0.bias", "multi_modal_projector.linear_1.bias"):
                    if k in key_to_tensor:
                        b = key_to_tensor[k]
                        break
                if w is not None:
                    assign_linear(linear_modules[0], w, b)
                    print("[debug] projector load: fallback to first layer for single Linear")
                    return True
                return False

            # 多层（如 mlp2x_gelu）：按常见命名逐一匹配
            assigned = 0
            layer_key_map = [
                # (idx, weight_keys, bias_keys)
                (
                    0,
                    ("model.mm_projector.0.weight", "multi_modal_projector.linear_1.weight"),
                    ("model.mm_projector.0.bias", "multi_modal_projector.linear_1.bias"),
                ),
                (
                    1,
                    ("model.mm_projector.2.weight", "multi_modal_projector.linear_2.weight"),
                    ("model.mm_projector.2.bias", "multi_modal_projector.linear_2.bias"),
                ),
            ]
            for idx, w_keys, b_keys in layer_key_map:
                if idx >= len(linear_modules):
                    continue
                w = next((key_to_tensor[k] for k in w_keys if k in key_to_tensor), None)
                b = next((key_to_tensor[k] for k in b_keys if k in key_to_tensor), None)
                if w is not None:
                    assign_linear(linear_modules[idx], w, b)
                    assigned += 1
            if assigned > 0:
                print(f"[debug] projector load: assigned {assigned} Linear layers")
                return True
            return False

        # 收集本地 Linear 模块（顺序即写入顺序）
        if isinstance(self.projector, nn.Linear):
            linear_modules = [self.projector]
        elif isinstance(self.projector, nn.Sequential):
            linear_modules = [m for m in self.projector if isinstance(m, nn.Linear)]
        else:
            raise RuntimeError(f"Unsupported projector type: {type(self.projector)}")

        found = False
        for sd in iter_state_dicts(weight_dir):
            if try_assign_from_keydict(sd):
                found = True
                break

        if not found:
            raise RuntimeError(
                "Projector weights not found in checkpoint. "
                "Expected keys like 'model.mm_projector.{0,2}.(weight|bias)' or "
                "'multi_modal_projector.linear_{1,2}.(weight|bias)' "
                "or 'model.mm_projector.(weight|bias)'."
            )

    def load_model(self, weight_dir):
        print(f"[debug] load vision model: {weight_dir}")
        vision_config = Mineru2QwenConfig.from_pretrained(weight_dir)

        self.vision_tower = build_vision_tower(vision_config)
        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)
        self.projector = build_vision_projector(vision_config)
        self._load_projector_weights(weight_dir)
        # 取配置参数传下去
        self.image_processor = Mineru2ImageProcessor(
            image_aspect_ratio=getattr(vision_config, "image_aspect_ratio", None),
            image_grid_pinpoints=getattr(vision_config, "image_grid_pinpoints", None),
        )

    def cuda(self):
        self.vision_tower = self.vision_tower.cuda()
        self.projector = self.projector.cuda()
        return self

    def forward(self, x) -> torch.Tensor:
        vision_out = self.vision_tower(x, output_hidden_states=True)
        hidden = vision_out.hidden_states[-1]
        # 对patch维度做平均池化，得到每视图一个向量
        pooled_per_view = hidden.mean(dim=1)
        proj = self.projector(pooled_per_view)
        return proj

    def encode(self, images: List[ImageItem]) -> Tuple[torch.Tensor, List[str], List[List[int]]]:
        print(f"[debug] mineru2_visual encode images {len(images)}")
        img_tensors: List[torch.Tensor] = []
        uuids: List[str] = []
        valid_id = 0
        valid_ids: List[List[int]] = []
        image_aspect_ratio = getattr(self.image_processor, "image_aspect_ratio", None)
        image_grid_pinpoints = getattr(self.image_processor, "image_grid_pinpoints", None)
        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data)).convert("RGB")
                if image_aspect_ratio == "pad":
                    image_proc = expand2square(image_data, tuple(int(x * 255) for x in self.image_processor.image_mean))
                    t = self.image_processor.preprocess(image_proc, return_tensors="pt")["pixel_values"]
                elif image_aspect_ratio and (image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio):
                    t = process_anyres_image(image_data, self.image_processor, image_grid_pinpoints)
                    if isinstance(t, np.ndarray):
                        t = torch.from_numpy(t)
                else:
                    t = self.image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"]

                if t.ndim == 5:
                    print(f"[debug] mineru2_visual reshape t.ndim: {t.ndim}, t.shape: {t.shape}")
                    t = t.view(-1, t.shape[-3], t.shape[-2], t.shape[-1])
                elif t.ndim == 3:
                    print(f"[debug] mineru2_visual unsqueeze t.ndim: {t.ndim}, t.shape: {t.shape}")
                    t = t.unsqueeze(0)
                # 在修改前记录 manager 分配的 token_num
                try:
                    print(f"[debug] mineru2_visual manager_token_num_before={img.token_num} uuid={img.uuid}")
                except Exception:
                    pass
                # 对齐实际 K 与期望 token_num
                expected_k = img.token_num if getattr(img, "token_num", None) is not None else None
                actual_k = t.shape[0]
                if expected_k is None or expected_k <= 0:
                    expected_k = actual_k
                    print(f"[debug] mineru2_visual expected_k_from_actual uuid={img.uuid} expected_k={expected_k}")
                if actual_k != expected_k:
                    if actual_k % expected_k == 0:
                        factor = actual_k // expected_k
                        print(
                            f"[debug] mineru2_visual down_aggregate uuid={img.uuid}"
                            f" actual_k={actual_k} expected_k={expected_k} factor={factor}"
                        )
                        t = t.view(expected_k, factor, t.shape[1], t.shape[2], t.shape[3]).mean(dim=1)
                    elif expected_k % actual_k == 0:
                        factor = expected_k // actual_k
                        print(
                            f"[debug] mineru2_visual up_repeat uuid={img.uuid}"
                            f" actual_k={actual_k} expected_k={expected_k} factor={factor}"
                        )
                        t = t.repeat_interleave(repeats=factor, dim=0)
                    else:
                        k = min(actual_k, expected_k)
                        print(
                            f"[debug] mineru2_visual fallback_slice uuid={img.uuid}"
                            f" actual_k={actual_k} expected_k={expected_k} k={k}"
                        )
                        if actual_k >= expected_k:
                            t = t[:expected_k]
                        else:
                            # pad by repeating last
                            pad = t[-1:].repeat(expected_k - actual_k, 1, 1, 1)
                            t = torch.cat([t, pad], dim=0)
                img_tensors.append(t)
                # 最终 K
                final_k = t.shape[0]
                img.token_num = final_k
                print(
                    f"[debug] mineru2_visual actual_k={actual_k} "
                    f"expected_k={expected_k} final_k={final_k} uuid={img.uuid}"
                )
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

            cur_num = (
                img_tensors[-1].shape[0]
                if isinstance(img_tensors[-1], torch.Tensor) and img_tensors[-1].dim() == 4
                else 1
            )
            valid_ids.append([valid_id, valid_id + cur_num])
            print(
                f"[debug] mineru2_visual valid_ids_append uuid={img.uuid}"
                f" range=({valid_id},{valid_id + cur_num}) cur_num={cur_num}"
            )
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None, [], []
        # 保证全部为4维后拼接
        img = torch.cat(img_tensors, dim=0)
        img = img.cuda()
        all_img_embeds = self.forward(img)
        print(f"[debug] mineru2_visual all_img_embeds.shape={tuple(all_img_embeds.shape)} " f"total_K={img.shape[0]}")

        return all_img_embeds, uuids, valid_ids
