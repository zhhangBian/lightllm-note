import re
import os
import json

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.server.core.objs import SamplingParams
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.internvl.layer_weights.pre_and_post_layer_weight import InternVLLlamaPreAndPostLayerWeight
from lightllm.models.internvl.img_process import get_image_patch

from ..mineru2_qwen.image_processing_mineru2 import Mineru2ImageProcessor
from .image_processing_mineru2 import get_anyres_image_grid_shape

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_TOKEN = "<image>"


class Mineru2QwenTokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer, model_cfg):
        super().__init__(tokenizer)

        self.image_token = model_cfg.get("image_token", IMG_TOKEN)
        self.img_token_index = model_cfg.get("image_token_index", 151646)

        self.image_start_tag = IMG_START_TOKEN
        self.image_start_id = tokenizer.convert_tokens_to_ids(self.image_start_tag)

        self.image_end_tag = IMG_END_TOKEN
        self.image_end_id = tokenizer.convert_tokens_to_ids(self.image_end_tag)

        if "text_config" in model_cfg:
            patch_size = model_cfg["vision_config"]["patch_size"]
            image_size = model_cfg["vision_config"]["image_size"]
        else:
            mm_vision_tower = model_cfg.get("mm_vision_tower", "google/siglip-so400m-patch14-384")
            if isinstance(mm_vision_tower, list):
                mm_vision_tower = mm_vision_tower[0]
            mm_vision_tower = mm_vision_tower.split("/")[-1]
            vision_tower_match = re.match(r"^siglip-(\w+)-patch(\d+)-(\d+)$", mm_vision_tower)
            patch_size = int(vision_tower_match.group(2))
            default_img_size = int(vision_tower_match.group(3))
            image_size = model_cfg.get("img_size", default_img_size)
            image_size = model_cfg.get("mm_image_size", image_size)

        self.image_processor = Mineru2ImageProcessor(
            image_aspect_ratio=(model_cfg.get("image_aspect_ratio", None)),
            image_grid_pinpoints=(model_cfg.get("image_grid_pinpoints", None)),
        )
        self.image_length = (image_size // patch_size) ** 2

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        raise NotImplementedError

    def get_image_token_length(self, img: ImageItem):
        # 切回 patch 序列：总token数 = 视图数 × 每视图patch数
        # 每视图patch数 = self.image_length = (image_size // patch_size) ** 2
        aspect = getattr(self.image_processor, "image_aspect_ratio", None)
        patch_len = int(self.image_length)
        try:
            if aspect and (aspect == "anyres" or (isinstance(aspect, str) and "anyres_max" in aspect)):
                crop_size = self.image_processor.crop_size["height"]
                grid_w, grid_h = get_anyres_image_grid_shape(
                    (img.image_w, img.image_h), self.image_processor.image_grid_pinpoints, crop_size
                )
                views = int(grid_w * grid_h + 1)
                token_num = views * patch_len
                print(
                    f"[debug] mineru2_tokenizer anyres img_size=({img.image_w},{img.image_h}) "
                    f"crop={crop_size} grid=({grid_w},{grid_h}) views={views}"
                    f" patch_len={patch_len} token_num={token_num}"
                )
                return token_num
            else:
                token_num = patch_len
                print(
                    f"[debug] mineru2_tokenizer non-anyres views=1 patch_len={patch_len}"
                    f" token_num={token_num} aspect={aspect}"
                )
                return token_num
        except Exception as e:
            # 兜底：按单视图返回
            token_num = patch_len
            print(f"[debug] mineru2_tokenizer token_num_fallback due to {e}, return {token_num}")
            return token_num

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None, add_special_tokens: bool = True):
        if multimodal_params is None:
            return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

        prompt = prompt.replace(IMG_START_TOKEN + IMG_END_TOKEN, IMG_TOKEN)

        origin_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        print(f"[debug] mineru2_tokenizer origin_ids={origin_ids}")

        # 单标记<image>：遇到img_token_index，直接展开K个占位token
        input_ids = []
        image_id = 0
        i = 0
        while i < len(origin_ids):
            tok = origin_ids[i]
            if tok == self.img_token_index:
                if image_id >= len(multimodal_params.images):
                    print("[warning] mineru2_tokenizer more <image> than provided images, keep literal token")
                    input_ids.append(tok)
                    i += 1
                    continue
                token_id = multimodal_params.images[image_id].token_id
                token_num = multimodal_params.images[image_id].token_num
                input_ids.extend(range(token_id, token_id + token_num))
                image_id += 1
                i += 1
            else:
                input_ids.append(tok)
                i += 1

        # 若有多余的图像对象，忽略并提示
        if image_id < len(multimodal_params.images):
            print(f"[warning] mineru2_tokenizer unused images: {len(multimodal_params.images) - image_id}")

        print(f"[debug] mineru2_tokenizer input_ids={input_ids}")
        return input_ids


@ModelRegistry("mineru2_qwen", is_multimodal=True)
class Mineru2QwenForCausalLM(Qwen2TpPartModel):
    # weight class
    # pre_and_post_weight_class = InternVLLlamaPreAndPostLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
