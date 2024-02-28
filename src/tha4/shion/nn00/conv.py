from typing import Optional, Union, Callable

from torch.nn import Conv2d, Module, Sequential, ConvTranspose2d

from tha4.shion.nn00.block_args import BlockArgs
from tha4.shion.nn00.linear_module_args import LinearModuleArgs, wrap_linear_module


def create_conv7(
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        linear_module_args: Optional[LinearModuleArgs] = None) -> Module:
    return wrap_linear_module(
        Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=bias),
        linear_module_args)


def create_conv3(in_channels: int,
                 out_channels: int,
                 bias: bool = False,
                 linear_module_args: Optional[LinearModuleArgs] = None) -> Module:
    return wrap_linear_module(
        Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        linear_module_args)


def create_conv1(
        in_channels: int, out_channels: int,
        bias: bool = False,
        linear_module_args: Optional[LinearModuleArgs] = None) -> Module:
    return wrap_linear_module(
        Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        linear_module_args)


def create_conv7_block(
        in_channels: int,
        out_channels: int,
        block_args: Optional[BlockArgs] = None) -> Module:
    if block_args is None:
        block_args = BlockArgs()
    return Sequential(
        create_conv7(
            in_channels,
            out_channels,
            bias=False,
            linear_module_args=block_args.linear_module_args),
        block_args.normalization_layer_factory.create(out_channels, affine=True),
        block_args.nonlinearity_factory.create())


def create_conv3_block(
        in_channels: int,
        out_channels: int,
        block_args: Optional[BlockArgs] = None) -> Module:
    if block_args is None:
        block_args = BlockArgs()
    return Sequential(
        create_conv7(
            in_channels,
            out_channels,
            bias=False,
            linear_module_args=block_args.linear_module_args),
        block_args.normalization_layer_factory.create(out_channels, affine=True),
        block_args.nonlinearity_factory.create())


def create_downsample_block(
        in_channels: int,
        out_channels: int,
        is_output_1x1: bool = False,
        block_args: Optional[BlockArgs] = None) -> Module:
    if block_args is None:
        block_args = BlockArgs()
    if is_output_1x1:
        return Sequential(
            wrap_linear_module(
                Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                block_args.linear_module_args),
            block_args.nonlinearity_factory.create())
    else:
        return Sequential(
            wrap_linear_module(
                Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                block_args.linear_module_args),
            block_args.normalization_layer_factory.create(out_channels, affine=True),
            block_args.nonlinearity_factory.create())


def create_upsample_block(
        in_channels: int,
        out_channels: int,
        block_args: Optional[BlockArgs] = None) -> Module:
    if block_args is None:
        block_args = BlockArgs()
    return Sequential(
        wrap_linear_module(
            ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            linear_module_args=block_args.linear_module_args),
        block_args.normalization_layer_factory.create(out_channels, affine=True),
        block_args.nonlinearity_factory.create())
