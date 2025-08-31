from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight

from .mineru2_visual import Mineru2VisionModel
from .layer_weights.pre_and_post_layer_weight import Mineru2QwenPreAndPostLayerWeight
from .layer_infer.pre_layer_infer import Mineru2QwenPreLayerInfer


@ModelRegistry.register("mineru2_qwen", "proxy")
class Mineru2QwenForCausalLM(Qwen2TpPartModel):
    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _init_some_layers(self):
        self.pre_and_post_weight = Mineru2QwenPreAndPostLayerWeight(
            self.tp_rank_, self.world_size_, self.data_type_, self.network_config_, self.mode
        )
        self.trans_err_layer_weight = Qwen2TransformerLayerWeight(
            self.tp_rank_, self.world_size_, self.data_type_, self.network_config_, self.mode
        )

        self.pre_layer_infer = Mineru2QwenPreLayerInfer(
            self.tp_rank_, self.world_size_, self.network_config_, self.mode
        )

    def _init_vis_config(self):
        self.visual_model = Mineru2VisionModel(self.config)
        self.image_token_index = self.config.image_token_index
