import math
import os
from enum import Enum
from typing import List

import numpy
import torch
from matplotlib import cm
from torch import Tensor
from torch.nn.functional import interpolate

from tha4.shion.base.image_util import save_numpy_image


class ImageSource(Enum):
    BATCH = 0
    OUTPUT = 1


class ImageType(Enum):
    COLOR = 0
    ALPHA = 1
    GRID_CHANGE = 2
    SIGMOID_LOGIT = 3


class SampleImageSpec:
    def __init__(self, image_source: ImageSource, index: int, image_type: ImageType):
        self.image_type = image_type
        self.index = index
        self.image_source = image_source


def torch_rgb_to_numpy_image(torch_image: Tensor, min_pixel_value=-1.0, max_pixel_value=1.0):
    assert torch_image.dim() == 3
    assert torch_image.shape[0] == 3
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    reshaped_image = torch_image.numpy().reshape(3, height * width).transpose().reshape(height, width, 3)
    numpy_image = (reshaped_image - min_pixel_value) / (max_pixel_value - min_pixel_value)
    return numpy_image


def torch_rgba_to_numpy_image(torch_image: Tensor, min_pixel_value=-1.0, max_pixel_value=1.0):
    assert torch_image.dim() == 3
    assert torch_image.shape[0] == 4
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    reshaped_image = torch_image.numpy().reshape(4, height * width).transpose().reshape(height, width, 4)
    numpy_image = (reshaped_image - min_pixel_value) / (max_pixel_value - min_pixel_value)
    numpy_image = numpy.clip(numpy_image, 0.0, 1.0)
    return numpy_image


def torch_grid_change_to_numpy_image(torch_image, num_channels=3):
    height = torch_image.shape[1]
    width = torch_image.shape[2]
    size_image = (torch_image[0, :, :] ** 2 + torch_image[1, :, :] ** 2).sqrt().view(height, width, 1).numpy()
    hsv = cm.get_cmap('hsv')
    angle_image = hsv(((torch.atan2(
        torch_image[0, :, :].view(height * width),
        torch_image[1, :, :].view(height * width)).view(height, width) + math.pi) / (2 * math.pi)).numpy()) * 3
    numpy_image = size_image * angle_image[:, :, 0:3]
    if num_channels == 3:
        return numpy_image
    elif num_channels == 4:
        return numpy.concatenate([numpy_image, numpy.ones_like(size_image)], axis=2)
    else:
        raise RuntimeError("Unsupported num_channels: " + str(num_channels))


class SampleImageSaver:
    def __init__(self,
                 image_size: int,
                 cell_size: int,
                 image_channels: int,
                 sample_image_specs: List[SampleImageSpec]):
        super().__init__()
        self.sample_image_specs = sample_image_specs
        self.cell_size = cell_size
        self.image_channels = image_channels
        self.image_size = image_size

    def save_sample_output_image(self, batch: List[Tensor], outputs: List[Tensor], file_name: str):
        num_cols = len(self.sample_image_specs)

        num_rows = batch[0].shape[0]
        output_image = numpy.zeros([self.cell_size * num_rows, self.cell_size * num_cols, self.image_channels])

        for image_index in range(num_rows):
            row = image_index
            start_row = row * self.cell_size

            for col in range(num_cols):
                spec = self.sample_image_specs[col]
                start_col = col * self.cell_size

                if spec.image_source == ImageSource.BATCH:
                    image = batch[spec.index][image_index].clone()
                else:
                    image = outputs[spec.index][image_index].clone()

                if spec.image_type == ImageType.COLOR:
                    c, h, w = image.shape
                    green_screen = torch.ones(3, h, w, device=image.device) * -1.0
                    green_screen[1, :, :] = 1.0
                    alpha = (image[3:4, :, :] + 1.0) * 0.5
                    image[0:3, :, :] = image[0:3, :, :] * alpha + green_screen * (1 - alpha)
                    image[3:4, :, :] = 1.0
                    image = image.detach().cpu()
                elif spec.image_type == ImageType.GRID_CHANGE:
                    image = image.detach().cpu()
                elif spec.image_type == ImageType.SIGMOID_LOGIT:
                    image = torch.sigmoid(image)
                    image = image.repeat(self.image_channels, 1, 1)
                    image = image * 2.0 - 1.0
                    image = image.detach().cpu()
                else:
                    if image.shape[0] == 1:
                        image = image.repeat(self.image_channels, 1, 1)
                    image = image * 2.0 - 1.0
                    image = image.detach().cpu()

                output_image[start_row:start_row + self.cell_size, start_col:start_col + self.cell_size, :] \
                    = self.convert_to_numpy_image(image)

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        save_numpy_image(output_image, file_name, save_straight_alpha=True)

    def save_sample_output_data(self,
                                batch: List[Tensor],
                                outputs: List[Tensor],
                                prefix: str,
                                examples_seen_so_far: int):
        file_name = "%s/sample_output_%010d.png" % (prefix, examples_seen_so_far)
        self.save_sample_output_image(batch, outputs, file_name)

    def convert_to_numpy_image(self, image: torch.Tensor):
        if self.cell_size != self.image_size:
            image = interpolate(image.unsqueeze(0), size=self.cell_size).squeeze(0)
        if image.shape[0] == 2:
            return torch_grid_change_to_numpy_image(image, num_channels=self.image_channels)
        elif self.image_channels == 3:
            return torch_rgb_to_numpy_image(image)
        else:
            return torch_rgba_to_numpy_image(image)
