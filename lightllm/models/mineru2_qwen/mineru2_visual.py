import torch
import torch.nn as nn
import re
import ast
import numpy as np
from functools import partial, reduce
from typing import Dict, Optional, Union

from PIL import Image
from transformers import (
    SiglipVisionConfig,
    SiglipVisionModel,
    BaseImageProcessor,
    BatchFeature,
)
from transformers.image_processing_utils import get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.utils import TensorType


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    original_width, original_height = original_size
    best_fit = (0, 0)
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def divide_to_patches(image, patch_size):
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)
    return patches


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        grid_pinpoints = [
            (i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)
        ]
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)  # type: ignore
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        patch_size = processor.crop_size["height"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        grid_pinpoints = [
            (i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)
        ]
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)  # type: ignore
    best_resolution = select_best_resolution(image.size, possible_resolutions)

    # image_padded = resize_and_pad_image(image, best_resolution)
    image_padded = image.resize(best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size["height"])

    image_original_resize = image.resize((processor.crop_size["height"], processor.crop_size["height"]))

    image_patches = [image_original_resize] + patches
    image_patches = [
        processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches
    ]
    return torch.stack(image_patches, dim=0)


class Mineru2ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: Optional[Dict[str, int]] = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[list] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints
        self.in_e2e_processing = False

    def _preprocess(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        return {"pixel_values": images}

    def _preprocess_end_to_end(self, images):
        image_aspect_ratio = self.image_aspect_ratio
        image_grid_pinpoints = self.image_grid_pinpoints
        assert image_aspect_ratio is not None
        assert image_grid_pinpoints is not None

        pixel_values = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = expand2square(image, tuple(int(x * 255) for x in self.image_mean))
                image = self._preprocess(image)["pixel_values"][0]
                pixel_values.append(image)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            for image in images:
                image_processed = process_anyres_image(image, self, self.image_grid_pinpoints)
                pixel_values.append(image_processed.numpy())
        else:
            pixel_values = self._preprocess(images)["pixel_values"]

        if isinstance(pixel_values, list) and all(x.shape == pixel_values[0].shape for x in pixel_values):
            pixel_values = np.stack(pixel_values, axis=0)

        # CAUTION: here used (height, width).
        image_sizes = [(image.height, image.width) for image in images]
        assert len(pixel_values) == len(image_sizes)

        return {"pixel_values": pixel_values, "image_sizes": image_sizes}

    def preprocess(
        self,
        images,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        if self.image_aspect_ratio is None or self.in_e2e_processing:
            data = self._preprocess(images)
        else:
            assert self.image_grid_pinpoints is not None
            self.in_e2e_processing = True
            try:
                data = self._preprocess_end_to_end(images)
            finally:
                self.in_e2e_processing = False

        return BatchFeature(data=data, tensor_type=return_tensors)


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()

        self.config = SiglipVisionConfig.from_pretrained(vision_tower)
        assert isinstance(self.config, SiglipVisionConfig)
        self.config.num_hidden_layers -= 1  # drop the last hidden layer
        self.config.vision_use_head = False

        self.vision_tower = SiglipVisionModel(self.config)
        self.vision_tower.requires_grad_(False)

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True
                )
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
            )
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size


def build_vision_tower(config):
    # ... (paste the full function code here)
    vision_tower = getattr(config, "mm_vision_tower", getattr(config, "vision_tower", ""))
    # In LightLLM, model path is handled by the framework, we can directly use the identifier
    if "siglip" in vision_tower.lower():
        return SiglipVisionTower(vision_tower)
    raise ValueError(f"Unknown vision tower: {vision_tower}")


def build_vision_projector(config):
    # ... (paste the full function code here)
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return nn.Identity()

    raise ValueError(f"Unknown projector type: {projector_type}")


class Mineru2VisionModel:
    def __init__(self, config):
        self.config = config
        self.vision_tower = build_vision_tower(config)
        self.mm_projector = build_vision_projector(config)
        self.image_processor = Mineru2ImageProcessor()

        # Set device and dtype for inference
        self.device = torch.device("cuda")
        self.dtype = config.torch_dtype
        self.vision_tower.to(device=self.device, dtype=self.dtype).eval()
        self.mm_projector.to(device=self.device, dtype=self.dtype).eval()

    @torch.no_grad()
    def get_vision_feature(self, images: list[Image.Image]):
        # Preprocess images using the specific logic of Mineru2
        processed_output = self.image_processor.preprocess(images, return_tensors="pt")
        pixel_values = processed_output["pixel_values"]
        image_sizes = processed_output.get("image_sizes", None)  # Important for anyres

        # If images are processed into a list of tensors (e.g., anyres), handle them one by one
        if isinstance(pixel_values, list):
            image_features_list = []
            for image_tensor in pixel_values:
                image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)
                features = self.vision_tower(image_tensor)
                projected_features = self.mm_projector(features)
                image_features_list.append(projected_features)

            # The return value expected by LightLLM should be a list of tensors and sizes
            return image_features_list, image_sizes

        # If images are batched into a single tensor
        else:
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            image_features = self.vision_tower(pixel_values)
            projected_features = self.mm_projector(image_features)

            # Since input is a list, output should also be a list of features for each image
            return [f for f in projected_features], image_sizes
