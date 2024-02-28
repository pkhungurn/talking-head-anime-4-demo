import math

import PIL.Image
import numpy
import torch
from matplotlib import cm
from tha4.shion.base.image_util import numpy_linear_to_srgb, pytorch_rgba_to_numpy_image, pytorch_rgb_to_numpy_image, \
    torch_linear_to_srgb


def grid_change_to_numpy_image(torch_image, num_channels=3):
    height = torch_image.shape[1]
    width = torch_image.shape[2]
    size_image = (torch_image[0, :, :] ** 2 + torch_image[1, :, :] ** 2).sqrt().view(height, width, 1).numpy()
    hsv = cm.get_cmap('hsv')
    angle_image = hsv(((torch.atan2(
        torch_image[0, :, :].view(height * width),
        torch_image[1, :, :].view(height * width)).view(height, width) + math.pi) / (2 * math.pi)).numpy()) * 3
    numpy_image = size_image * angle_image[:, :, 0:3]
    rgb_image = numpy_linear_to_srgb(numpy_image)
    if num_channels == 3:
        return rgb_image
    elif num_channels == 4:
        return numpy.concatenate([rgb_image, numpy.ones_like(size_image)], axis=2)
    else:
        raise RuntimeError("Unsupported num_channels: " + str(num_channels))


def resize_PIL_image(pil_image, size=(256, 256)):
    w, h = pil_image.size
    d = min(w, h)
    r = ((w - d) // 2, (h - d) // 2, (w + d) // 2, (h + d) // 2)
    return pil_image.resize(size, resample=PIL.Image.LANCZOS, box=r)


def convert_output_image_from_torch_to_numpy(output_image):
    if output_image.shape[2] == 2:
        h, w, c = output_image.shape
        numpy_image = torch.transpose(output_image.reshape(h * w, c), 0, 1).reshape(c, h, w)
    elif output_image.shape[0] == 4:
        numpy_image = pytorch_rgba_to_numpy_image(output_image)
    elif output_image.shape[0] == 3:
        numpy_image = pytorch_rgb_to_numpy_image(output_image)
    elif output_image.shape[0] == 1:
        c, h, w = output_image.shape
        alpha_image = torch.cat([output_image.repeat(3, 1, 1) * 2.0 - 1.0, torch.ones(1, h, w)], dim=0)
        numpy_image = pytorch_rgba_to_numpy_image(alpha_image)
    elif output_image.shape[0] == 2:
        numpy_image = grid_change_to_numpy_image(output_image, num_channels=4)
    else:
        raise RuntimeError("Unsupported # image channels: %d" % output_image.shape[0])
    numpy_image = numpy.uint8(numpy.rint(numpy_image * 255.0))
    return numpy_image


def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)
