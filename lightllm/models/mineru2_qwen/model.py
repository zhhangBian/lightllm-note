from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.qwen2.layer_weights.pre_and_post_layer_weight import Qwen2PreAndPostLayerWeight
from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight


@ModelRegistry("mineru2_qwen", is_multimodal=True)
class Mineru2QwenForCausalLM(Qwen2TpPartModel):
    # weight class
    pre_and_post_weight_class = Qwen2PreAndPostLayerWeight
    transformer_weight_class = Qwen2TransformerLayerWeight

    def __init__(self, kvargs):
        super().__init__(kvargs)
