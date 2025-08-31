import torch
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb


class Mineru2QwenPreLayerInfer(LlamaPreLayerInfer):
    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)

    def _multimodal_forward(self, input_ids, image_embeds, **kwargs):
        if image_embeds is None:
            return self.model.embed_tokens(input_ids)

        input_mask = input_ids == self.model.image_token_index

        image_token_len = [img_emb.shape[0] for img_emb in image_embeds]

        concat_image_features = torch.cat(image_embeds, dim=0)

        final_input_embeds = multimodal_emb(
            self.model.embed_tokens.weight, input_ids, input_mask, concat_image_features, image_token_len
        )
        return final_input_embeds

    def context_forward(self, input_ids, batch_info, **kwargs):
        image_embeds = batch_info.get("image_embeds", None)

        input_embeds = self._multimodal_forward(input_ids, image_embeds)
        return input_embeds

    def token_forward(self, input_ids, batch_info, **kwargs):
        input_embeds = self.model.embed_tokens(input_ids)
        return input_embeds
