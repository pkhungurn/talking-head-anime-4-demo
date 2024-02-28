from typing import List

import torch
from torch import Tensor, zero_
from torch.nn import Module, Conv2d

from tha4.shion.core.module_factory import ModuleFactory
from tha4.nn.image_processing_util import GridChangeApplier
from tha4.nn.common.unet import UnetArgs, AttentionBlockArgs, UnetWithFirstConvAddition


class Upscaler02Args:
    def __init__(self,
                 image_size: int,
                 image_channels: int,
                 num_pose_parameters: int,
                 unet_args: UnetArgs):
        assert unet_args.in_channels == (
            image_channels
        )
        assert unet_args.out_channels == (
                image_channels +  # direct
                2 +  # warp
                1  # alpha
        )
        assert unet_args.cond_input_channels == num_pose_parameters
        self.image_channels = image_channels
        self.image_size = image_size
        self.num_pose_parameters = num_pose_parameters
        self.unet_args = unet_args


def apply_color_change(alpha, color_change, image: Tensor) -> Tensor:
    return color_change * alpha + image * (1 - alpha)


class Upscaler02(Module):
    def __init__(self, args: Upscaler02Args):
        super().__init__()
        self.args = args
        self.body = UnetWithFirstConvAddition(args.unet_args)
        self.grid_change_applier = GridChangeApplier()
        self.coarse_image_conv = Conv2d(
            args.image_channels + args.image_channels + 2,
            args.unet_args.model_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        with torch.no_grad():
            zero_(self.coarse_image_conv.weight)
            zero_(self.coarse_image_conv.bias)

    def check_image(self, image: torch.Tensor):
        assert len(image.shape) == 4
        assert image.shape[1] == self.args.image_channels
        assert image.shape[2] == self.args.image_size
        assert image.shape[3] == self.args.image_size

    def forward(self,
                rest_image: torch.Tensor,
                coarse_posed_image: torch.Tensor,
                coarse_grid_change: torch.Tensor,
                pose: torch.Tensor) -> List[Tensor]:
        self.check_image(rest_image)
        self.check_image(coarse_posed_image)

        assert len(pose.shape) == 2
        assert rest_image.shape[0] == pose.shape[0]
        assert coarse_posed_image.shape[0] == pose.shape[0]
        assert coarse_grid_change.shape[0] == pose.shape[0]
        assert coarse_grid_change.shape[1] == 2
        assert coarse_grid_change.shape[2] == self.args.image_size
        assert coarse_grid_change.shape[3] == self.args.image_size
        assert pose.shape[1] == self.args.num_pose_parameters

        warped_image = self.grid_change_applier.apply(coarse_grid_change, rest_image)

        t = torch.zeros(rest_image.shape[0], 1, device=rest_image.device)
        feature = torch.cat([coarse_posed_image, warped_image, coarse_grid_change], dim=1)
        first_conv_addition = self.coarse_image_conv(feature)

        body_output = self.body(rest_image, t, pose, first_conv_addition)

        direct = body_output[:, 0:self.args.image_channels, :, :]
        grid_change = body_output[:, self.args.image_channels:self.args.image_channels + 2, :, :]
        alpha = torch.sigmoid(body_output[:, self.args.image_channels + 2:self.args.image_channels + 3, :, :])
        warped = self.grid_change_applier.apply(grid_change, rest_image)
        merged = apply_color_change(alpha, direct, warped)

        return [
            merged,
            alpha,
            warped,
            grid_change,
            direct
        ]

    INDEX_MERGED = 0
    INDEX_ALPHA = 1
    INDEX_WARPED = 2
    INDEX_GRID_CHANGE = 3
    INDEX_DIRECT = 4


class Upscaler02Factory(ModuleFactory):
    def __init__(self, args: Upscaler02Args):
        self.args = args

    def create(self) -> Module:
        return Upscaler02(self.args)
