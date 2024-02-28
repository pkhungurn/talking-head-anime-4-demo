from typing import Optional, List

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import affine_grid

from tha4.shion.core.module_factory import ModuleFactory
from tha4.nn.siren.vanilla.siren import SirenArgs, Siren


class SirenFaceMorpher00Args:
    def __init__(self,
                 image_size: int,
                 image_channels: int,
                 pose_size: int,
                 siren_args: SirenArgs):
        assert siren_args.in_channels == pose_size + 2
        assert siren_args.out_channels == image_channels
        assert not siren_args.use_tanh

        self.siren_args = siren_args
        self.pose_size = pose_size
        self.image_size = image_size
        self.image_channels = image_channels


class SirenFaceMorpher00(Module):
    def __init__(self, args: SirenFaceMorpher00Args):
        super().__init__()
        self.args = args
        self.siren = Siren(self.args.siren_args)

    def forward(self, pose: Tensor, position: Optional[Tensor] = None) -> Tensor:
        n, p = pose.shape[0], pose.shape[1]
        device = pose.device

        if position is None:
            h, w = self.args.image_size, self.args.image_size
            identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device).unsqueeze(0)
            position = affine_grid(identity, [1, 1, h, w], align_corners=False) \
                .view(1, h * w, 2)
            position = torch.transpose(position, dim0=1, dim1=2).view(1, 2, h, w) \
                .repeat(n, 1, 1, 1)

        h, w = position.shape[2], position.shape[3]
        pose_image = pose.view(n, p, 1, 1).repeat(1, 1, h, w)

        siren_input = torch.cat([position, pose_image], dim=1)

        return self.siren.forward(siren_input)


class SirenFaceMorpher00Factory(ModuleFactory):
    def __init__(self, args: SirenFaceMorpher00Args):
        self.args = args

    def create(self) -> Module:
        return SirenFaceMorpher00(self.args)
