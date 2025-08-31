import torch
from lightllm.models.qwen2.layer_weights.pre_and_post_layer_weight import Qwen2PreAndPostLayerWeight


class Mineru2QwenPreAndPostLayerWeight(Qwen2PreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)

    def load_hf_weights(self, weights):
        # First, load all the standard Qwen2 weights (embed_tokens, lm_head, norm)
        # by calling the parent class's method.
        super().load_hf_weights(weights)

        # Now, add the logic to load the multimodal projector weights.
        # The names must match the keys in your model's state_dict.
        if "model.mm_projector.0.weight" in weights:
            # Assuming a simple MLP projector like nn.Sequential(nn.Linear(...), nn.GELU(), nn.Linear(...))
            # You may need to adjust the keys based on your actual projector structure.

            # Example for a 2-layer MLP projector
            proj_w1 = weights["model.mm_projector.0.weight"]
            proj_b1 = weights["model.mm_projector.0.bias"]
            proj_w2 = weights["model.mm_projector.2.weight"]
            proj_b2 = weights["model.mm_projector.2.bias"]

            self.mm_projector_w1_ = self._cuda(proj_w1)
            self.mm_projector_b1_ = self._cuda(proj_b1)
            self.mm_projector_w2_ = self._cuda(proj_w2)
            self.mm_projector_b2_ = self._cuda(proj_b2)

    def _verify_params(self):
        # It's good practice to verify the loaded parameters.
        super()._verify_params()
        assert hasattr(self, "mm_projector_w1_"), "mm_projector weights not loaded"

    def _copy_to_model_layers(self, model_layers):
        # Copy the standard weights first
        super()._copy_to_model_layers(model_layers)

        # Copy our custom weights to the mm_projector in the model
        # Note: self.model_layers_ is set in the parent class
        if hasattr(self, "mm_projector_w1_"):
            self.model_layers_["mm_projector.0.weight"].copy_(self.mm_projector_w1_)
            self.model_layers_["mm_projector.0.bias"].copy_(self.mm_projector_b1_)
            self.model_layers_["mm_projector.2.weight"].copy_(self.mm_projector_w2_)
            self.model_layers_["mm_projector.2.bias"].copy_(self.mm_projector_b2_)
