import os
from enum import Enum
from typing import List, Dict

import PIL.Image
import numpy
import torch
from tha4.shion.base.dataset.util import get_indexed_batch
from tha4.shion.base.image_util import pytorch_rgb_to_numpy_image, pytorch_rgba_to_numpy_image
from tha4.shion.core.cached_computation import TensorCachedComputationFunc, ComputationState
from tha4.shion.core.training.sample_output_protocol import SampleOutputProtocol
from tha4.image_util import grid_change_to_numpy_image
from torch.nn import Module
from torch.nn.functional import interpolate
from torch.utils.data import Dataset


class ImageType(Enum):
    COLOR = 0
    ALPHA = 1
    GRID_CHANGE = 2
    SIGMOID_LOGIT = 3


class SampleImageSpec:
    def __init__(self, value_func: TensorCachedComputationFunc, image_type: ImageType):
        self.value_func = value_func
        self.image_type = image_type


class SampleImageSaver:
    def __init__(self,
                 cell_size: int,
                 image_channels: int,
                 sample_image_specs: List[SampleImageSpec]):
        super().__init__()
        self.sample_image_specs = sample_image_specs
        self.cell_size = cell_size
        self.image_channels = image_channels

    def save_sample_output_data(self,
                                state: ComputationState,
                                prefix: str,
                                examples_seen_so_far: int):
        num_cols = len(self.sample_image_specs)
        num_rows = state.batch[0].shape[0]
        output_image = numpy.zeros([self.cell_size * num_rows, self.cell_size * num_cols, self.image_channels])

        for col in range(num_cols):
            spec = self.sample_image_specs[col]
            images = spec.value_func(state)
            start_col = col * self.cell_size

            for image_index in range(num_rows):
                image = images[image_index].clone().detach()
                row = image_index
                start_row = row * self.cell_size
                if spec.image_type == ImageType.COLOR:
                    c, h, w = image.shape
                    green_screen = torch.ones(3, h, w, device=image.device) * -1.0
                    green_screen[1, :, :] = 1.0
                    alpha = (image[3:4, :, :] + 1.0) * 0.5
                    image[0:3, :, :] = image[0:3, :, :] * alpha + green_screen * (1 - alpha)
                    image[3:4, :, :] = 1.0
                    image = image.cpu()
                elif spec.image_type == ImageType.GRID_CHANGE:
                    image = image.cpu()
                elif spec.image_type == ImageType.SIGMOID_LOGIT:
                    image = torch.sigmoid(image)
                    image = image.repeat(self.image_channels, 1, 1)
                    image = image * 2.0 - 1.0
                    image = image.cpu()
                elif spec.image_type == ImageType.ALPHA:
                    if image.shape[0] == 1:
                        image = image.repeat(self.image_channels, 1, 1)
                    image = image * 2.0 - 1.0
                    image = image.cpu()
                else:
                    raise RuntimeError(f"Unsupported image type: {spec.image_type}")

                output_image[start_row:start_row + self.cell_size, start_col:start_col + self.cell_size, :] \
                    = self.convert_to_numpy_image(image)

        file_name = "%s/sample_output_%010d.png" % (prefix, examples_seen_so_far)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        if self.image_channels == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(output_image * 255.0)), mode=mode)
        pil_image.save(file_name)
        print("Saved %s" % file_name)

    def convert_to_numpy_image(self, image: torch.Tensor):
        image_size = image.shape[-1]
        if self.cell_size != image_size:
            image = interpolate(image.unsqueeze(0), size=self.cell_size).squeeze(0)
        if image.shape[0] == 2:
            return grid_change_to_numpy_image(image, num_channels=self.image_channels)
        elif self.image_channels == 3:
            return pytorch_rgb_to_numpy_image(image)
        else:
            return pytorch_rgba_to_numpy_image(image)


class GeneralSampleOutputProtocol(SampleOutputProtocol):
    def __init__(self,
                 sample_image_specs: List[SampleImageSpec],
                 num_images: int = 8,
                 cell_size: int = 256,
                 image_channels: int = 4,
                 examples_per_sample_output: int = 5000,
                 random_seed: int = 1203040687):
        super().__init__()
        self.num_images = num_images
        self.random_seed = random_seed
        self.examples_per_sample_output = examples_per_sample_output
        self.sample_image_saver = SampleImageSaver(cell_size, image_channels, sample_image_specs)

    def get_examples_per_sample_output(self) -> int:
        return self.examples_per_sample_output

    def get_random_seed(self) -> int:
        return self.random_seed

    def get_sample_output_data(self, validation_dataset: Dataset, device: torch.device) -> dict:
        example_indices = torch.randint(0, len(validation_dataset), (self.num_images,))
        example_indices = [example_indices[i].item() for i in range(self.num_images)]
        batch = get_indexed_batch(validation_dataset, example_indices, device)
        return {'batch': batch}

    def save_sample_output_data(self,
                                modules: Dict[str, Module],
                                accumulated_modules: Dict[str, Module],
                                sample_output_data: dict, prefix: str,
                                examples_seen_so_far: int,
                                device: torch.device):
        for key in modules:
            modules[key].train(False)
        batch = sample_output_data['batch']
        state = ComputationState(modules, accumulated_modules, batch, {})
        self.sample_image_saver.save_sample_output_data(state, prefix, examples_seen_so_far)
