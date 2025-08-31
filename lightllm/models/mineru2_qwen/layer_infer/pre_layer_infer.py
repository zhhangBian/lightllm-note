import torch
from lightllm.models.qwen2.layer_infer.pre_layer_infer import Qwen2PreLayerInfer
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb


class Mineru2QwenPreLayerInfer(Qwen2PreLayerInfer):
    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)

    def _multimodal_forward(self, input_ids, image_embeds, **kwargs):
        # This method overrides the parent's behavior to handle image embeddings.

        if image_embeds is None:
            # If no images are present, fall back to the standard text-only embedding lookup.
            return self.model.embed_tokens(input_ids)

        # When images are present, use the optimized Triton kernel for fusion.
        input_mask = input_ids == self.model.image_token_index

        # The kernel needs the total number of image tokens to correctly allocate memory.
        # This assumes image_embeds is a list of tensors, one for each image in the batch.
        image_token_len = [img_emb.shape[0] for img_emb in image_embeds]

        # Concatenate all image features into a single tensor for the kernel.
        concat_image_features = torch.cat(image_embeds, dim=0)

        # Call the kernel
        final_input_embeds = multimodal_emb(
            self.model.embed_tokens.weight, input_ids, input_mask, concat_image_features, image_token_len
        )
        return final_input_embeds

    # We override the main context_forward to ensure our _multimodal_forward is called
    def context_forward(self, input_ids, batch_info, **kwargs):
        # Extract image_embeds passed from the Router
        image_embeds = batch_info.get("image_embeds", None)

        input_embeds = self._multimodal_forward(input_ids, image_embeds)
        return input_embeds

    # We also override token_forward for the decoding steps
    def token_forward(self, input_ids, batch_info, **kwargs):
        # In decoding, there are no new images, so we just do a standard embedding lookup.
        # The prompt with fused embeddings is already cached in the KV cache.
        input_embeds = self.model.embed_tokens(input_ids)
        return input_embeds
