import os

import PIL.Image
import numpy
import torch
from matplotlib import pyplot
from torch import Tensor


def numpy_srgb_to_linear(x):
    x = numpy.clip(x, 0.0, 1.0)
    return numpy.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def numpy_linear_to_srgb(x):
    x = numpy.clip(x, 0.0, 1.0)
    return numpy.where(x <= 0.003130804953560372, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def numpy_alpha_devide(rgb, a, epsilon=1e-5):
    aaa = numpy.repeat(a, 3, axis=2)
    aaa_prime = aaa + numpy.where(numpy.abs(aaa) < epsilon, epsilon, 0.0)
    return numpy.where(numpy.abs(aaa) < epsilon, 0.0, rgb / aaa_prime)


def torch_srgb_to_linear(x: torch.Tensor):
    x = torch.clip(x, 0.0, 1.0)
    return torch.where(torch.le(x, 0.04045), x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def torch_linear_to_srgb(x):
    x = torch.clip(x, 0.0, 1.0)
    return torch.where(torch.le(x, 0.003130804953560372), x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def numpy_image_linear_to_srgb(image):
    assert image.shape[2] == 3 or image.shape[2] == 4
    if image.shape[2] == 3:
        return numpy_linear_to_srgb(image)
    else:
        height, width, _ = image.shape
        rgb_image = numpy_linear_to_srgb(image[:, :, 0:3])
        a_image = image[:, :, 3:4]
        return numpy.concatenate((rgb_image, a_image), axis=2)


def numpy_image_srgb_to_linear(image):
    assert image.shape[2] == 3 or image.shape[2] == 4
    if image.shape[2] == 3:
        return numpy_srgb_to_linear(image)
    else:
        height, width, _ = image.shape
        rgb_image = numpy_srgb_to_linear(image[:, :, 0:3])
        a_image = image[:, :, 3:4]
        return numpy.concatenate((rgb_image, a_image), axis=2)


def pytorch_rgb_to_numpy_image(torch_image: Tensor, min_pixel_value=-1.0, max_pixel_value=1.0):
    assert torch_image.dim() == 3
    assert torch_image.shape[0] == 3
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    reshaped_image = torch_image.numpy().reshape(3, height * width).transpose().reshape(height, width, 3)
    numpy_image = (reshaped_image - min_pixel_value) / (max_pixel_value - min_pixel_value)
    return numpy_linear_to_srgb(numpy_image)


def pytorch_rgba_to_numpy_image_greenscreen(torch_image: Tensor,
                                            min_pixel_value=-1.0,
                                            max_pixel_value=1.0,
                                            include_alpha=False):
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    numpy_image = (torch_image.numpy().reshape(4, height * width).transpose().reshape(height, width,
                                                                                      4) - min_pixel_value) \
                  / (max_pixel_value - min_pixel_value)
    rgb_image = numpy_linear_to_srgb(numpy_image[:, :, 0:3])
    a_image = numpy_image[:, :, 3]
    rgb_image[:, :, 0:3] = rgb_image[:, :, 0:3] * a_image.reshape(a_image.shape[0], a_image.shape[1], 1)
    rgb_image[:, :, 1] = rgb_image[:, :, 1] + (1 - a_image)

    if not include_alpha:
        return rgb_image
    else:
        return numpy.concatenate((rgb_image, numpy.ones_like(numpy_image[:, :, 3:4])), axis=2)


def pytorch_rgba_to_numpy_image(
        torch_image: Tensor,
        min_pixel_value=-1.0,
        max_pixel_value=1.0,
        perform_linear_to_srb: bool = True):
    assert torch_image.dim() == 3
    assert torch_image.shape[0] == 4
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    reshaped_image = torch_image.numpy().reshape(4, height * width).transpose().reshape(height, width, 4)
    numpy_image = (reshaped_image - min_pixel_value) / (max_pixel_value - min_pixel_value)
    if perform_linear_to_srb:
        rgb_image = numpy_linear_to_srgb(numpy_image[:, :, 0:3])
    else:
        rgb_image = numpy.clip(numpy_image[:, :, 0:3], 0.0, 1.0)
    a_image = numpy.clip(numpy_image[:, :, 3], 0.0, 1.0)
    rgba_image = numpy.concatenate((rgb_image, a_image.reshape(height, width, 1)), axis=2)
    return rgba_image


def pil_image_has_transparency(pil_image):
    if pil_image.info.get("transparency", None) is not None:
        return True
    if pil_image.mode == "P":
        transparent = pil_image.info.get("transparency", -1)
        for _, index in pil_image.getcolors():
            if index == transparent:
                return True
    elif pil_image.mode == "RGBA":
        extrema = pil_image.getextrema()
        if extrema[3][0] < 255:
            return True

    return False


def extract_numpy_image_from_PIL_image(pil_image, scale=2.0, offset=-1.0,
                                       premultiply_alpha=True,
                                       perform_srgb_to_linear=True):
    has_alpha = pil_image_has_transparency(pil_image)
    if has_alpha and pil_image.mode != 'RGBA':
        pil_image = pil_image.convert("RGBA")
    if not has_alpha and pil_image.mode != 'RGB':
        pil_image = pil_image.convert("RGB")
    if has_alpha:
        num_channel = 4
    else:
        num_channel = 3
    image_width = pil_image.width
    image_height = pil_image.height

    raw_image = numpy.asarray(pil_image, dtype=numpy.float32)
    image = (raw_image / 255.0).reshape(image_height, image_width, num_channel)
    if perform_srgb_to_linear:
        image[:, :, 0:3] = numpy_srgb_to_linear(image[:, :, 0:3])
        # Premultiply alpha
    if has_alpha and premultiply_alpha:
        image[:, :, 0:3] = image[:, :, 0:3] * image[:, :, 3:4]
    return image * scale + offset


def extract_numpy_image_from_PIL_image_with_pytorch_layout(pil_image, scale=2.0, offset=-1.0,
                                                           premultiply_alpha=True,
                                                           perform_srgb_to_linear=True):
    numpy_image = extract_numpy_image_from_PIL_image(
        pil_image, scale, offset, premultiply_alpha, perform_srgb_to_linear)
    image_height, image_width, num_channel = numpy_image.shape
    image = numpy_image \
        .reshape(image_height * image_width, num_channel) \
        .transpose() \
        .reshape(num_channel, image_height, image_width)
    return image


def extract_numpy_image_from_filelike_with_pytorch_layout(file, scale=2.0, offset=-1.0, premultiply_alpha=True):
    try:
        pil_image = PIL.Image.open(file)
    except Exception as e:
        raise RuntimeError(file)
    return extract_numpy_image_from_PIL_image_with_pytorch_layout(pil_image, scale, offset, premultiply_alpha)


def extract_numpy_image_from_filelike(file, scale=1.0, offset=0.0,
                                      premultiply_alpha=True,
                                      perform_srgb_to_linear: bool = True):
    try:
        pil_image = PIL.Image.open(file)
    except Exception as e:
        raise RuntimeError(file)
    return extract_numpy_image_from_PIL_image(pil_image, scale, offset, premultiply_alpha, perform_srgb_to_linear)


def extract_pytorch_image_from_filelike(file, scale=2.0, offset=-1.0, premultiply_alpha=True,
                                        perform_srgb_to_linear=True):
    try:
        pil_image = PIL.Image.open(file)
    except Exception as e:
        raise RuntimeError(file)
    image = extract_numpy_image_from_PIL_image_with_pytorch_layout(pil_image, scale, offset, premultiply_alpha,
                                                                   perform_srgb_to_linear)
    return torch.from_numpy(image).float()


def extract_pytorch_image_from_PIL_image(pil_image, scale=2.0, offset=-1.0, premultiply_alpha=True,
                                         perform_srgb_to_linear=True):
    image = extract_numpy_image_from_PIL_image_with_pytorch_layout(
        pil_image, scale, offset, premultiply_alpha, perform_srgb_to_linear)
    return torch.from_numpy(image).float()


def convert_pytorch_image_to_zero_to_one_numpy_image(
        torch_image: torch.Tensor,
        scale: float = 2.0,
        offset: float = -1.0):
    torch_image = (torch_image - offset) / scale
    torch_image = torch.permute(torch_image, (1, 2, 0))
    numpy_image = torch_image.cpu().numpy()
    return numpy_image


def convert_zero_to_one_numpy_image_to_PIL_image(
        numpy_image, use_straight_alpha=True, perform_linear_to_srgb=True):
    if numpy_image.shape[2] == 4:
        rgb_image = numpy_image[:, :, 0:3]
        a_image = numpy.clip(numpy_image[:, :, 3:4], 0.0, 1.0)
        if use_straight_alpha:
            rgb_image = numpy_alpha_devide(rgb_image, a_image)
        if perform_linear_to_srgb:
            rgb_image = numpy_linear_to_srgb(rgb_image)
        else:
            rgb_image = numpy.clip(rgb_image, 0.0, 1.0)
        new_numpy_image = numpy.concatenate((rgb_image, a_image), axis=2)
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(new_numpy_image * 255.0)), mode='RGBA')
    else:
        if perform_linear_to_srgb:
            numpy_image = numpy_linear_to_srgb(numpy_image)
        else:
            numpy_image = numpy.clip(numpy_image, 0.0, 1.0)
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGB')
    return pil_image


def save_numpy_image(numpy_image, file_name: str, save_straight_alpha=True, perform_linear_to_srgb=True):
    pil_image = convert_zero_to_one_numpy_image_to_PIL_image(numpy_image, save_straight_alpha, perform_linear_to_srgb)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    pil_image.save(file_name)


def resize_PIL_image(pil_image, size=(256, 256)):
    w, h = pil_image.size
    d = min(w, h)
    r = ((w - d) // 2, (h - d) // 2, (w + d) // 2, (h + d) // 2)
    return pil_image.resize(size, resample=PIL.Image.LANCZOS, box=r)