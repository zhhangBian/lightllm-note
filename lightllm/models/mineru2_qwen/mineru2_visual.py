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
        # 方案A：使用配置减层并按该配置实例化模型，再加载权重（忽略不匹配尺寸）
        cfg = SiglipVisionConfig.from_pretrained(vt_path)
        old_layers = getattr(cfg, "num_hidden_layers", None)
        cfg.num_hidden_layers = max(0, cfg.num_hidden_layers - 1)
        cfg.vision_use_head = False
        model = SiglipVisionModel.from_pretrained(vt_path, config=cfg, ignore_mismatched_sizes=True)
        try:
            actual_layers = len(model.vision_model.encoder.layers)  # type: ignore[attr-defined]
        except Exception:
            actual_layers = None
        new_cfg_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
        print(f"[debug] siglip_layers planA old={old_layers} new_cfg={new_cfg_layers} actual_module={actual_layers}")
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
        # 运行时形状与精度/设备检查
        try:
            print(f"[debug] mineru2_visual.forward x.shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
        except Exception:
            pass
        vision_out = self.vision_tower(x, output_hidden_states=True)
        hiddens = vision_out.hidden_states
        # hidden_states 数量与 config 层数的关系（一般为 num_layers + 1）
        try:
            cfg_layers = getattr(getattr(self.vision_tower, "config", None), "num_hidden_layers", None)
            eff_layers = len(hiddens) - 1 if isinstance(hiddens, (list, tuple)) else None
            print(
                f"[debug] mineru2_visual.hidden_states len={len(hiddens)}"
                f" cfg_layers={cfg_layers} eff_layers={eff_layers}"
            )
        except Exception:
            pass
        # 对齐ref的“减一层”语义：优先使用倒数第二层；若不可用则回退最后一层
        try:
            chosen_idx = -2 if isinstance(hiddens, (list, tuple)) and len(hiddens) >= 2 else -1
            feat = hiddens[chosen_idx]
            print(f"[debug] mineru2_visual.select_layer idx={chosen_idx} feat.shape={tuple(feat.shape)}")
        except Exception:
            feat = hiddens[-2] if isinstance(hiddens, (list, tuple)) and len(hiddens) >= 2 else hiddens[-1]
        # 切回 patch 序列特征：去除 CLS（若存在），按序列过 projector，再展平为 (views*patch, hidden)
        patch_side = self.vision_tower.config.image_size // self.vision_tower.config.patch_size
        patch_len = patch_side * patch_side
        if feat.shape[1] == patch_len + 1:
            feat = feat[:, 1:, :]
            print(f"[debug] mineru2_visual.drop_cls patch_len={patch_len} feat_no_cls.shape={tuple(feat.shape)}")
        proj_seq = self.projector(feat)
        try:
            print(f"[debug] mineru2_visual.projector_seq_out shape={tuple(proj_seq.shape)} (views, patch, hidden)")
        except Exception:
            pass
        proj = proj_seq.reshape(-1, proj_seq.shape[-1])
        try:
            print(f"[debug] mineru2_visual.projector_flat_out shape={tuple(proj.shape)} (views*patch, hidden)")
        except Exception:
            pass
        return proj

    def encode(self, images: List[ImageItem]) -> Tuple[torch.Tensor, List[str], List[List[int]]]:
        print(f"[debug] mineru2_visual encode images {len(images)}")
        img_tensors: List[torch.Tensor] = []
        uuids: List[str] = []
        valid_id = 0
        valid_ids: List[List[int]] = []
        image_aspect_ratio = getattr(self.image_processor, "image_aspect_ratio", None)
        image_grid_pinpoints = getattr(self.image_processor, "image_grid_pinpoints", None)
        # 每视图 patch_len（例如 384/14=27, 27^2=729）
        patch_side = self.vision_tower.config.image_size // self.vision_tower.config.patch_size
        patch_len = patch_side * patch_side
        print(f"[debug] mineru2_visual.patch_len={patch_len} (side={patch_side})")
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
                # 在修改前记录 manager 分配的 token_num（可能是视图数或视图*patch数）
                try:
                    print(f"[debug] mineru2_visual manager_token_num_before={img.token_num} uuid={img.uuid}")
                except Exception:
                    pass
                # 对齐实际视图数 K 与期望 token（可能是 K 或 K*patch_len）
                expected_token = img.token_num if getattr(img, "token_num", None) is not None else None
                actual_k = t.shape[0]
                if expected_token is None or expected_token <= 0:
                    expected_views = actual_k
                    print(
                        f"[debug] mineru2_visual expected_views_from_actual uuid={img.uuid}"
                        f" expected_views={expected_views}"
                    )
                else:
                    if expected_token >= patch_len and expected_token % patch_len == 0:
                        expected_views = expected_token // patch_len
                        print(
                            f"[debug] mineru2_visual expected_views_from_tokens uuid={img.uuid}"
                            f" expected_token={expected_token} patch_len={patch_len} expected_views={expected_views}"
                        )
                    else:
                        expected_views = expected_token
                        print(
                            f"[debug] mineru2_visual expected_views_interpret_as_views uuid={img.uuid}"
                            f" expected_views={expected_views}"
                        )
                if actual_k != expected_views:
                    if actual_k % expected_views == 0:
                        factor = actual_k // expected_views
                        print(
                            f"[debug] mineru2_visual down_aggregate uuid={img.uuid}"
                            f" actual_k={actual_k} expected_views={expected_views} factor={factor}"
                        )
                        t = t.view(expected_views, factor, t.shape[1], t.shape[2], t.shape[3]).mean(dim=1)
                    elif expected_views % actual_k == 0:
                        factor = expected_views // actual_k
                        print(
                            f"[debug] mineru2_visual up_repeat uuid={img.uuid}"
                            f" actual_k={actual_k} expected_views={expected_views} factor={factor}"
                        )
                        t = t.repeat_interleave(repeats=factor, dim=0)
                    else:
                        k = min(actual_k, expected_views)
                        print(
                            f"[debug] mineru2_visual fallback_slice uuid={img.uuid}"
                            f" actual_k={actual_k} expected_views={expected_views} k={k}"
                        )
                        if actual_k >= expected_views:
                            t = t[:expected_views]
                        else:
                            # pad by repeating last
                            pad = t[-1:].repeat(expected_views - actual_k, 1, 1, 1)
                            t = torch.cat([t, pad], dim=0)
                img_tensors.append(t)
                # 最终视图数 K
                final_views = t.shape[0]
                # 对齐 patch 序列后的总 token 数
                img.token_num = final_views * patch_len
                print(
                    f"[debug] mineru2_visual actual_k={actual_k} expected_views={expected_views}"
                    f" final_views={final_views} final_token_num={img.token_num} uuid={img.uuid}"
                )
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

            # 本图对应的 token 数（视图 * patch_len）
            if isinstance(img_tensors[-1], torch.Tensor) and img_tensors[-1].dim() == 4:
                cur_num = img_tensors[-1].shape[0] * patch_len
            else:
                cur_num = patch_len
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
        print(
            f"[debug] mineru2_visual all_img_embeds.shape={tuple(all_img_embeds.shape)}"
            f" total_tokens={img.shape[0] * patch_len}"
        )

        return all_img_embeds, uuids, valid_ids
