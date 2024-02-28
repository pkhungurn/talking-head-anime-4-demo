from typing import List, Optional, Callable

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential, Conv2d
from torch.nn.functional import affine_grid, interpolate

from tha4.shion.core.module_factory import ModuleFactory
from tha4.shion.nn00.initialization_funcs import HeInitialization
from tha4.nn.image_processing_util import GridChangeApplier
from tha4.nn.siren.vanilla.siren import SineLinearLayer


class SirenMorpherLevelArgs:
    def __init__(self,
                 image_size: int,
                 intermediate_channels: int,
                 num_sine_layers: int):
        assert num_sine_layers >= 2
        self.image_size = image_size
        self.num_sine_layers = num_sine_layers
        self.intermediate_channels = intermediate_channels


class SirenMorpher03Args:
    def __init__(self,
                 image_size: int,
                 image_channels: int,
                 pose_size: int,
                 level_args: List[SirenMorpherLevelArgs],
                 init_func: Optional[Callable[[Module], Module]] = None):
        assert len(level_args) >= 2
        if init_func is None:
            init_func = HeInitialization()
        self.image_size = image_size
        self.init_func = init_func
        self.level_args = level_args
        self.pose_size = pose_size
        self.image_channels = image_channels


class SirenMorpher03(Module):
    def __init__(self, args: SirenMorpher03Args):
        super().__init__()
        self.args = args

        self.siren_layers = ModuleList()

        for i in range(len(args.level_args)):
            level_args = args.level_args[i]

            layers = []

            if i == 0:
                layers.append(SineLinearLayer(
                    in_channels=args.pose_size + 2,
                    out_channels=level_args.intermediate_channels,
                    is_first=True))
            else:
                layers.append(SineLinearLayer(
                    in_channels=level_args.intermediate_channels + args.pose_size + 2,
                    out_channels=level_args.intermediate_channels,
                    is_first=False))

            for j in range(1, level_args.num_sine_layers - 1):
                layers.append(SineLinearLayer(
                    in_channels=level_args.intermediate_channels,
                    out_channels=level_args.intermediate_channels,
                    is_first=False))

            if i == len(args.level_args) - 1:
                out_channels = level_args.intermediate_channels
            else:
                out_channels = args.level_args[i + 1].intermediate_channels
            layers.append(SineLinearLayer(
                in_channels=level_args.intermediate_channels,
                out_channels=out_channels,
                is_first=False))

            self.siren_layers.append(Sequential(*layers))

        self.last_linear = args.init_func(Conv2d(
            args.level_args[-1].intermediate_channels,
            args.image_channels + 2 + 1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True))

        self.grid_change_applier = GridChangeApplier()

    def get_position_grid(self, n: int, image_size: int, device: torch.device):
        h, w = image_size, image_size
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device).unsqueeze(0)
        position = affine_grid(identity, [1, 1, h, w], align_corners=False) \
            .view(1, h * w, 2)
        position = torch.transpose(position, dim0=1, dim1=2).view(1, 2, h, w) \
            .repeat(n, 1, 1, 1)
        return position

    def get_pose_image(self, pose: Tensor, image_size: int):
        n, p = pose.shape[0], pose.shape[1]
        h, w = image_size, image_size
        pose_image = pose.view(n, p, 1, 1).repeat(1, 1, h, w)
        return pose_image

    def forward(self, image: Tensor, pose: Tensor) -> List[Tensor]:
        n = pose.shape[0]
        device = pose.device

        x = None
        for i in range(len(self.args.level_args)):
            args = self.args.level_args[i]
            position_and_pose = torch.cat([
                self.get_position_grid(n, args.image_size, device),
                self.get_pose_image(pose, args.image_size)
            ], dim=1)
            if i == 0:
                x = self.siren_layers[i].forward(position_and_pose)
            else:
                x = interpolate(x, size=(args.image_size, args.image_size), mode='bilinear')
                x = torch.cat([x, position_and_pose], dim=1)
                x = self.siren_layers[i].forward(x)

        siren_output = self.last_linear(x)

        grid_change = siren_output[:, 0:2, :, :]
        alpha = siren_output[:, 2:3, :, :]
        color_change = siren_output[:, 3:, :, :]
        warped_image = self.grid_change_applier.apply(grid_change, image, align_corners=False)
        blended_image = (1 - alpha) * warped_image + alpha * color_change

        return [
            blended_image,
            alpha,
            color_change,
            warped_image,
            grid_change
        ]

    INDEX_BLENDED_IMAGE = 0
    INDEX_ALPHA = 1
    INDEX_COLOR_CHANGE = 2
    INDEX_WARPED_IMAGE = 3
    INDEX_GRID_CHANGE = 4


class SirenMorpher03Factory(ModuleFactory):
    def __init__(self, args: SirenMorpher03Args):
        self.args = args

    def create(self):
        return SirenMorpher03(self.args)
