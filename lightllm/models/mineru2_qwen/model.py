from lightllm.models.qwen2_vl.model import Qwen2VLTpPartModel
from lightllm.models.registry import ModelRegistry


@ModelRegistry("mineru2_qwen", is_multimodal=True)
class Mineru2QwenForCausalLM(Qwen2VLTpPartModel):
    def __init__(self, kvargs):
        super().__init__(kvargs)
