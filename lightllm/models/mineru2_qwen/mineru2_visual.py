import re

from typing import List, Tuple
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    CLIPVisionModel,
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


def build_vision_tower(weight_dir: str, config: Mineru2QwenConfig):
    vision_tower = getattr(config, "mm_vision_tower", getattr(config, "vision_tower", ""))
    model_path = os.path.join(weight_dir, vision_tower)
    if not os.path.exists(model_path):
        model_path = vision_tower

    if "clip" in vision_tower.lower():
        return CLIPVisionModel.from_pretrained(model_path)
    elif "siglip" in vision_tower.lower():
        cfg = SiglipVisionConfig.from_pretrained(model_path)
        cfg.num_hidden_layers = max(0, cfg.num_hidden_layers - 1)
        cfg.vision_use_head = False
        model = SiglipVisionModel.from_pretrained(model_path, config=cfg, ignore_mismatched_sizes=True)
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
        def assign_linear(linear: nn.Linear, w: torch.Tensor = None, b: torch.Tensor = None):
            if w is not None:
                linear.weight.data.copy_(w.to(dtype=linear.weight.dtype))
            if b is not None and linear.bias is not None:
                linear.bias.data.copy_(b.to(dtype=linear.bias.dtype))

        projector_weight_path = os.path.join(weight_dir, "model.safetensors")

        if isinstance(self.projector, nn.Linear):
            linear_modules = [self.projector]
        elif isinstance(self.projector, nn.Sequential):
            linear_modules = [m for m in self.projector if isinstance(m, nn.Linear)]
        else:
            raise RuntimeError(f"Unsupported projector type: {type(self.projector)}")

        try:
            with safe_open(projector_weight_path, framework="pt", device="cpu") as sf:
                sd = {k: sf.get_tensor(k) for k in sf.keys()}
        except Exception as e:
            raise RuntimeError(f"Failed to read projector weights: {projector_weight_path} due to {e}")

        # load projector weights
        layer_key_map = [
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
            w = next((sd[k] for k in w_keys if k in sd), None)
            b = next((sd[k] for k in b_keys if k in sd), None)
            if w is not None:
                assign_linear(linear_modules[idx], w, b)

        # load vision tower weights
        vt_prefix = "model.vision_tower.vision_tower."
        vt_sd = {k[len(vt_prefix) :]: v for k, v in sd.items() if k.startswith(vt_prefix)}
        if not vt_sd:
            logger.warning("vision_tower weights not found in checkpoint or skipped; keep pretrained weights")
            return

        try:
            self.vision_tower.load_state_dict(vt_sd, strict=False)
        except Exception as e:
            logger.warning(f"vision_tower load_state_dict failed (strict=False): {e}")

    def load_model(self, weight_dir):
        vision_config = Mineru2QwenConfig.from_pretrained(weight_dir)

        self.vision_tower = build_vision_tower(weight_dir, vision_config)
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
        hiddens = vision_out.hidden_states

        # 对齐ref的“减一层”语义：优先使用倒数第二层；若不可用则回退最后一层
        try:
            chosen_idx = -2 if isinstance(hiddens, (list, tuple)) and len(hiddens) >= 2 else -1
            feat = hiddens[chosen_idx]
        except Exception:
            feat = hiddens[-2] if isinstance(hiddens, (list, tuple)) and len(hiddens) >= 2 else hiddens[-1]
        # 切回 patch 序列特征：去除 CLS（若存在），按序列过 projector，再展平为 (views*patch, hidden)
        patch_side = self.vision_tower.config.image_size // self.vision_tower.config.patch_size
        patch_len = patch_side * patch_side
        if feat.shape[1] == patch_len + 1:
            feat = feat[:, 1:, :]
        proj_seq = self.projector(feat)

        proj = proj_seq.reshape(-1, proj_seq.shape[-1])
        return proj

    def encode(self, images: List[ImageItem]) -> Tuple[torch.Tensor, List[str], List[List[int]]]:
        img_tensors: List[torch.Tensor] = []
        uuids: List[str] = []
        valid_id = 0
        valid_ids: List[List[int]] = []
        image_aspect_ratio = getattr(self.image_processor, "image_aspect_ratio", None)
        image_grid_pinpoints = getattr(self.image_processor, "image_grid_pinpoints", None)
        # 每视图 patch_len（例如 384/14=27, 27^2=729）
        patch_side = self.vision_tower.config.image_size // self.vision_tower.config.patch_size
        patch_len = patch_side * patch_side

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
                    t = t.view(-1, t.shape[-3], t.shape[-2], t.shape[-1])
                elif t.ndim == 3:
                    t = t.unsqueeze(0)

                # 对齐实际视图数 K 与期望 token（可能是 K 或 K*patch_len）
                expected_token = img.token_num if getattr(img, "token_num", None) is not None else None
                actual_k = t.shape[0]
                if expected_token is None or expected_token <= 0:
                    expected_views = actual_k
                else:
                    if expected_token >= patch_len and expected_token % patch_len == 0:
                        expected_views = expected_token // patch_len
                    else:
                        expected_views = expected_token
                if actual_k != expected_views:
                    if actual_k % expected_views == 0:
                        factor = actual_k // expected_views
                        t = t.view(expected_views, factor, t.shape[1], t.shape[2], t.shape[3]).mean(dim=1)
                    elif expected_views % actual_k == 0:
                        factor = expected_views // actual_k
                        t = t.repeat_interleave(repeats=factor, dim=0)
                    else:
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
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

            # 本图对应的 token 数（视图 * patch_len）
            if isinstance(img_tensors[-1], torch.Tensor) and img_tensors[-1].dim() == 4:
                cur_num = img_tensors[-1].shape[0] * patch_len
            else:
                cur_num = patch_len
            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None, [], []
        # 保证全部为4维后拼接
        img = torch.cat(img_tensors, dim=0)
        img = img.cuda()
        all_img_embeds = self.forward(img)

        return all_img_embeds, uuids, valid_ids
