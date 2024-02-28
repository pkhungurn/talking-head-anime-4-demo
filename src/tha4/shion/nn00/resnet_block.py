from typing import Optional

import torch
from torch.nn import Module, Sequential, Parameter

from tha4.shion.nn00.block_args import BlockArgs
from tha4.shion.nn00.conv import create_conv1, create_conv3


class ResnetBlock(Module):
    def __init__(self,
                 num_channels: int,
                 is1x1: bool = False,
                 use_scale_parameter: bool = False,
                 block_args: Optional[BlockArgs] = None):
        super().__init__()
        if block_args is None:
            block_args = BlockArgs()
        self.use_scale_parameter = use_scale_parameter
        if self.use_scale_parameter:
            self.scale = Parameter(torch.zeros(1))
        if is1x1:
            self.resnet_path = Sequential(
                create_conv1(
                    num_channels,
                    num_channels,
                    bias=True,
                    linear_module_args=block_args.linear_module_args),
                block_args.nonlinearity_factory.create(),
                create_conv1(
                    num_channels,
                    num_channels,
                    bias=True,
                    linear_module_args=block_args.linear_module_args))
        else:
            self.resnet_path = Sequential(
                create_conv3(
                    num_channels,
                    num_channels,
                    bias=False,
                    linear_module_args=block_args.linear_module_args),
                block_args.normalization_layer_factory.create(num_channels, affine=True),
                block_args.nonlinearity_factory.create(),
                create_conv3(
                    num_channels,
                    num_channels,
                    bias=False,
                    linear_module_args=block_args.linear_module_args),
                block_args.normalization_layer_factory.create(num_channels, affine=True))

    def forward(self, x):
        if self.use_scale_parameter:
            return x + self.scale * self.resnet_path(x)
        else:
            return x + self.resnet_path(x)
