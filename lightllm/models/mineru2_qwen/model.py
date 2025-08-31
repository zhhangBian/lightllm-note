from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.registry import ModelRegistry
from .configuration_mineru2 import Mineru2QwenConfig


@ModelRegistry("mineru2_qwen", is_multimodal=True)
class Mineru2QwenForCausalLM(Qwen2TpPartModel):
    # a new config class is not necessary
    config_class = Mineru2QwenConfig

    def __init__(self, kvargs):
        super().__init__(kvargs)
