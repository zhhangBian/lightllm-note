import torch
from lightllm.models.qwen2.layer_weights.pre_and_post_layer_weight import Qwen2PreAndPostLayerWeight


class Mineru2QwenPreAndPostLayerWeight(Qwen2PreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)

        if "model.mm_projector.0.weight" in weights:
            proj_w1 = weights["model.mm_projector.0.weight"]
            proj_b1 = weights["model.mm_projector.0.bias"]
            proj_w2 = weights["model.mm_projector.2.weight"]
            proj_b2 = weights["model.mm_projector.2.bias"]

            self.mm_projector_w1_ = self._cuda(proj_w1)
            self.mm_projector_b1_ = self._cuda(proj_b1)
            self.mm_projector_w2_ = self._cuda(proj_w2)
            self.mm_projector_b2_ = self._cuda(proj_b2)

    def _verify_params(self):
        super()._verify_params()
        assert hasattr(self, "mm_projector_w1_"), "mm_projector weights not loaded"

    def _copy_to_model_layers(self, model_layers):
        super()._copy_to_model_layers(model_layers)

        if hasattr(self, "mm_projector_w1_"):
            self.model_layers_["mm_projector.0.weight"].copy_(self.mm_projector_w1_)
            self.model_layers_["mm_projector.0.bias"].copy_(self.mm_projector_b1_)
            self.model_layers_["mm_projector.2.weight"].copy_(self.mm_projector_w2_)
            self.model_layers_["mm_projector.2.bias"].copy_(self.mm_projector_b2_)
