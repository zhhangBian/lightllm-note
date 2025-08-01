import json
import numpy as np
import unicodedata
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.qwen.model import QWenTpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from lightllm.server.core.objs import SamplingParams
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from typing import List, Optional, Union
from transformers.utils import TensorType, logging
from lightllm.models.qwen2_vl.flashattention_infer_struct import Qwen2VLFlashAttentionStateInfo
from lightllm.common.build_utils import repair_config
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo
from lightllm.models.qwen2_vl.layer_infer.transformer_layer_infer import Qwen2VLTransformerLayerInfer

import torch
from PIL import Image
from .vision_process import smart_resize
from lightllm.utils.envs_utils import enable_env_vars, get_env_start_args
from lightllm.models.qwen2.layer_weights import transformer_layer_weight, pre_and_post_layer_weight
from lightllm.models.qwen2.model import Qwen2TpPartModel
import os

# Warp of the origal tokenizer
class QWen2VLTokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer=None, image_processor=None, **kwargs):
        super().__init__(tokenizer)
        self.image_processor = image_processor
        self.image_start_id = kwargs["model_cfg"]["vision_start_token_id"]
        self.image_end_id = kwargs["model_cfg"]["vision_end_token_id"]
        self.image_token_id = kwargs["model_cfg"]["image_token_id"]

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        raise NotImplementedError

    def get_image_token_length(self, img: ImageItem):
        width = img.image_w
        height = img.image_h
        resized_height, resized_width = smart_resize(height=height, width=width)
        self.patch_size = self.image_processor.image_processor.patch_size
        self.merge_size = self.image_processor.image_processor.merge_size
        grid_t = 1
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        merge_length = self.merge_size ** 2
        self.token_num = (grid_t * grid_h * grid_w) // merge_length
        self.image_length = self.token_num
        return self.image_length

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):

        origin_ids = self.tokenizer.encode(prompt)

        # <img><image_pad></img> -> <img></img>
        origin_ids = [token for token in origin_ids if token != self.image_token_id]
        # <img></img> --> <img>id,id+1...id+num</img>
        input_ids = []
        image_id = 0
        start_idx = 0
        while True:
            try:
                start_idx = origin_ids.index(self.image_start_id, start_idx)
                if start_idx + 1 >= len(origin_ids):
                    break
                if origin_ids[start_idx + 1] == self.image_end_id:
                    input_ids.extend(origin_ids[: start_idx + 1])
                    token_id = multimodal_params.images[image_id].token_id
                    token_num = multimodal_params.images[image_id].token_num
                    input_ids.extend(range(token_id, token_id + token_num))
                    input_ids.append(self.image_end_id)
                    origin_ids = origin_ids[start_idx + 2 :]
                    start_idx = 0
                    image_id += 1
                else:
                    raise ValueError("image token error")
            except ValueError:
                break
        input_ids.extend(origin_ids[start_idx:])
        return input_ids


@ModelRegistry(["qwen2_vl", "qwen2_5_vl"], is_multimodal=True)
class Qwen2VLTpPartModel(Qwen2TpPartModel):

    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen2VLTransformerLayerInfer

    infer_state_class = Qwen2VLInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_inferstate_cls(self):
        if get_env_start_args().enable_fa3:
            self.infer_state_class = Qwen2VLFlashAttentionStateInfo

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
