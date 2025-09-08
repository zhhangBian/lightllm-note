import re
import os
import json

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.server.core.objs import SamplingParams
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.qwen2_vl.vision_process import smart_resize

from ..mineru2_qwen.image_processing_mineru2 import Mineru2ImageProcessor


class Mineru2QwenTokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer, model_cfg):
        super().__init__(tokenizer)
        self.image_token = model_cfg.get("image_token", "<image>")
        # for llava-v1.5-7b-hf model
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
        # (image_size // patch_size) ** 2: (384 // 14) ** 2 = 729
        self.image_length = (image_size // patch_size) ** 2
        self.skip_start = model_cfg.get("skip_start", True)

        self.image_processor = Mineru2ImageProcessor()

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        raise NotImplementedError

    def get_image_token_length(self, img: ImageItem):
        return self.image_length

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None, add_special_tokens: bool = True):

        origin_ids = self.tokenizer.encode(prompt)

        return origin_ids


@ModelRegistry("mineru2_qwen", is_multimodal=True)
class Mineru2QwenForCausalLM(Qwen2TpPartModel):
    def __init__(self, kvargs):
        super().__init__(kvargs)
