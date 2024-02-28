from typing import List

import torch
from torch import Tensor
from torch.nn import Module

from tha4.shion.core.module_factory import ModuleFactory
from tha4.nn.image_processing_util import GridChangeApplier
from tha4.nn.common.unet import UnetArgs, Unet, AttentionBlockArgs


def apply_color_change(alpha, color_change, image: Tensor) -> Tensor:
    return color_change * alpha + image * (1 - alpha)


class Morpher00Args:
    def __init__(self,
                 image_size: int,
                 image_channels: int,
                 num_pose_parameters: int,
                 unet_args: UnetArgs):
        assert unet_args.in_channels == image_channels
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


class Morpher00(Module):
    def __init__(self, args: Morpher00Args):
        super().__init__()
        self.args = args
        self.body = Unet(args.unet_args)
        self.grid_change_applier = GridChangeApplier()

    def forward(self, image: torch.Tensor, pose: torch.Tensor) -> List[Tensor]:
        assert len(image.shape) == 4
        assert image.shape[1] == self.args.image_channels
        assert image.shape[2] == self.args.image_size
        assert image.shape[3] == self.args.image_size
        assert len(pose.shape) == 2
        assert image.shape[0] == pose.shape[0]
        assert pose.shape[1] == self.args.num_pose_parameters

        t = torch.zeros(image.shape[0], 1, device=image.device)
        body_output = self.body(image, t, pose)
        direct = body_output[:, 0:self.args.image_channels, :, :]
        grid_change = body_output[:, self.args.image_channels:self.args.image_channels + 2, :, :]
        alpha = torch.sigmoid(body_output[:, self.args.image_channels + 2:self.args.image_channels + 3, :, :])

        warped = self.grid_change_applier.apply(grid_change, image)
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


class Morpher00Factory(ModuleFactory):
    def __init__(self, args: Morpher00Args):
        self.args = args

    def create(self) -> Module:
        return Morpher00(self.args)
