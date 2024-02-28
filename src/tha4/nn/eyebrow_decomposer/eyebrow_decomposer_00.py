from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module

from tha4.nn.common.poser_encoder_decoder_00 import PoserEncoderDecoder00Args, PoserEncoderDecoder00
from tha4.nn.image_processing_util import apply_color_change
from tha4.shion.core.module_factory import ModuleFactory
from tha4.nn.nonlinearity_factory import ReLUFactory
from tha4.nn.normalization import InstanceNorm2dFactory
from tha4.nn.util import BlockArgs


class EyebrowDecomposer00Args(PoserEncoderDecoder00Args):
    def __init__(self,
                 image_size: int = 128,
                 image_channels: int = 4,
                 start_channels: int = 64,
                 bottleneck_image_size=16,
                 num_bottleneck_blocks=6,
                 max_channels: int = 512,
                 block_args: Optional[BlockArgs] = None):
        super().__init__(
            image_size,
            image_channels,
            image_channels,
            0,
            start_channels,
            bottleneck_image_size,
            num_bottleneck_blocks,
            max_channels,
            block_args)


class EyebrowDecomposer00(Module):
    def __init__(self, args: EyebrowDecomposer00Args):
        super().__init__()
        self.args = args
        self.body = PoserEncoderDecoder00(args)
        self.background_layer_alpha = self.args.create_alpha_block()
        self.background_layer_color_change = self.args.create_color_change_block()
        self.eyebrow_layer_alpha = self.args.create_alpha_block()
        self.eyebrow_layer_color_change = self.args.create_color_change_block()

    def forward(self, image: Tensor, *args) -> List[Tensor]:
        feature = self.body(image)[0]

        background_layer_alpha = self.background_layer_alpha(feature)
        background_layer_color_change = self.background_layer_color_change(feature)
        background_layer_1 = apply_color_change(background_layer_alpha, background_layer_color_change, image)

        eyebrow_layer_alpha = self.eyebrow_layer_alpha(feature)
        eyebrow_layer_color_change = self.eyebrow_layer_color_change(feature)
        eyebrow_layer = apply_color_change(eyebrow_layer_alpha, image, eyebrow_layer_color_change)

        return [
            eyebrow_layer,  # 0
            eyebrow_layer_alpha,  # 1
            eyebrow_layer_color_change,  # 2
            background_layer_1,  # 3
            background_layer_alpha,  # 4
            background_layer_color_change,  # 5
        ]

    EYEBROW_LAYER_INDEX = 0
    EYEBROW_LAYER_ALPHA_INDEX = 1
    EYEBROW_LAYER_COLOR_CHANGE_INDEX = 2
    BACKGROUND_LAYER_INDEX = 3
    BACKGROUND_LAYER_ALPHA_INDEX = 4
    BACKGROUND_LAYER_COLOR_CHANGE_INDEX = 5
    OUTPUT_LENGTH = 6


class EyebrowDecomposer00Factory(ModuleFactory):
    def __init__(self, args: EyebrowDecomposer00Args):
        super().__init__()
        self.args = args

    def create(self) -> Module:
        return EyebrowDecomposer00(self.args)
