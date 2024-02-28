import math
from enum import Enum
from typing import Optional, List

import torch
from torch import zero_, Tensor
from torch.nn import Module, GroupNorm, Sequential, SiLU, Conv2d, AvgPool2d, Linear, Dropout, ModuleList
from torch.nn.functional import interpolate

from tha4.shion.core.module_factory import ModuleFactory


class Identity(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class IdentityFactory(ModuleFactory):
    def create(self) -> Module:
        return Identity()


def init_to_zero(module: Module):
    with torch.no_grad():
        zero_(module.weight)
        zero_(module.bias)
    return module


class Upsample(Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, use_conv: bool = False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        if use_conv or in_channels != out_channels:
            self.postprocess = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.postprocess = Identity()

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        return self.postprocess(interpolate(x, scale_factor=2, mode="nearest"))


class Downsample(Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, use_conv: bool = False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        if use_conv or in_channels != out_channels:
            self.op = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        return self.op(x)


def GroupNorm32(channels):
    return GroupNorm(min(32, channels), channels)


class SamplingMode(Enum):
    SAME_RESOLUTION = 0
    UPSAMPLING = 1
    DOWNSAMPING = 2


class ResBlockArgs:
    def __init__(self,
                 dropout_prob: float,
                 use_cond0: bool = True,
                 use_cond1: bool = False,
                 init_conditioned_residual_to_zero: bool = False,
                 use_conv_on_skip_connection: bool = False):
        assert not use_cond1 or use_cond0
        self.use_conv_on_skip_connection = use_conv_on_skip_connection
        self.use_cond1 = use_cond1
        self.use_cond0 = use_cond0
        self.init_conditioned_residual_to_zero = init_conditioned_residual_to_zero
        self.dropout_prob = dropout_prob


def apply_scaleshift(x: Tensor, scaleshift: Tensor, condition_bias: float = 1.0) -> Tensor:
    assert len(scaleshift.shape) == 2
    assert len(x.shape) == 4
    assert x.shape[0] == scaleshift.shape[0]
    assert 2 * x.shape[1] == scaleshift.shape[1]
    scaleshift = scaleshift.reshape(scaleshift.shape[0], scaleshift.shape[1], 1, 1)
    scale, shift = torch.chunk(scaleshift, 2, dim=1)
    return x * (condition_bias + scale) + shift


class ResBlock(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cond0_channels: Optional[int] = None,
                 cond1_channels: Optional[int] = None,
                 sampling_mode: SamplingMode = SamplingMode.SAME_RESOLUTION,
                 dropout_prob: float = 0.1,
                 condition_bias: float = 1.0):
        super().__init__()
        assert cond0_channels is not None or cond1_channels is None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sampling_mode = sampling_mode
        self.cond0_channels = cond0_channels
        self.cond1_channels = cond1_channels
        self.condition_bias = condition_bias

        if sampling_mode == SamplingMode.UPSAMPLING:
            self.x_resample = Upsample(in_channels)
            self.h_resample = Upsample(in_channels)
        elif sampling_mode == SamplingMode.DOWNSAMPING:
            self.x_resample = Downsample(in_channels)
            self.h_resample = Downsample(in_channels)
        else:
            self.x_resample = Identity()
            self.h_resample = Identity()

        self.nonlinear = SiLU()

        # Layers before conditioning
        self.norm0 = GroupNorm32(in_channels)
        self.conv0 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Conditioning layers
        if cond0_channels is not None:
            self.cond0_layers = Sequential(
                SiLU(),
                Linear(cond0_channels, 2 * out_channels))
            self.norm1 = GroupNorm32(out_channels)
            self.dropout = Dropout(dropout_prob)
            self.conv1 = init_to_zero(Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if cond1_channels is not None:
            self.cond1_layers = Sequential(
                SiLU(),
                Linear(cond0_channels, 2 * out_channels))

        # Skip layer
        if in_channels == out_channels:
            self.skip = Identity()
        else:
            self.skip = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor, cond0: Optional[Tensor] = None, cond1: Optional[Tensor] = None) -> Tensor:
        assert self.cond0_channels is None or cond0 is not None
        assert self.cond1_channels is None or cond1 is not None

        h = self.conv0(self.h_resample(self.nonlinear(self.norm0(x))))
        if self.cond0_channels is not None:
            h = self.norm1(h)
            h = apply_scaleshift(h, self.cond0_layers(cond0), self.condition_bias)
            if self.cond1_channels is not None:
                h = apply_scaleshift(h, self.cond1_layers(cond1), self.condition_bias)
            h = self.conv1(self.dropout(self.nonlinear(h)))
        return self.skip(self.x_resample(x)) + h


class AttentionBlockArgs:
    def __init__(self,
                 num_heads: Optional[int] = 1,
                 num_head_channels: Optional[int] = None,
                 use_new_attention_order: bool = False):
        self.use_new_attention_order = use_new_attention_order
        self.num_head_channels = num_head_channels
        self.num_heads = num_heads


def qkv_attention_legacy(qkv: torch.Tensor, num_heads: int):
    assert len(qkv.shape) == 3
    B, W, L = qkv.shape
    H = num_heads
    assert W % (3 * H) == 0
    C = W // (3 * H)
    q, k, v = qkv.reshape(B * H, C * 3, L).split(C, dim=1)
    scale = 1.0 / math.sqrt(math.sqrt(C))
    weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
    weight = torch.softmax(weight, dim=-1)
    output = torch.einsum("bts,bcs->bct", weight, v)
    return output.reshape(B, H * C, L)


def qkv_attention(qkv: torch.Tensor, num_heads: int):
    B, W, L = qkv.shape
    H = num_heads
    assert W % (3 * H) == 0
    C = W // (3 * H)
    q, k, v = qkv.chunk(3, dim=1)
    scale = 1.0 / math.sqrt(math.sqrt(C))
    weight = torch.einsum("bct,bcs->bts", (q * scale).view(B * H, C, L), (k * scale).view(B * H, C, L))
    weight = torch.softmax(weight, dim=-1)
    output = torch.einsum("bts,bcs->bct", weight, v.reshape(B * H, C, L))
    return output.reshape(B, H * C, L)


class AttentionBlock(Module):
    def __init__(self,
                 num_channels: int,
                 args: AttentionBlockArgs):
        super().__init__()
        self.use_new_attention_order = args.use_new_attention_order

        if args.num_head_channels is None:
            assert args.num_heads is not None
            assert num_channels % args.num_heads == 0
            self.num_heads = args.num_heads
            self.num_head_channels = num_channels // self.num_heads
        elif args.num_heads is None:
            assert args.num_head_channels is not None
            assert num_channels % args.num_head_channels == 0
            self.num_heads = num_channels // args.num_head_channels
            self.num_head_channels = args.num_head_channels

        self.norm = GroupNorm32(num_channels)
        self.qkv = Conv2d(num_channels, 3 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv = Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0)
        with torch.no_grad():
            zero_(self.conv.weight)
            zero_(self.conv.bias)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 4
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3 * C, H * W)
        if self.use_new_attention_order:
            h = qkv_attention(qkv, self.num_heads)
        else:
            h = qkv_attention_legacy(qkv, self.num_heads)
        h = self.conv(h.reshape(B, C, H, W))
        return x + h


class Arity3To1(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor, y: Optional[Tensor] = None, z: Optional[Tensor] = None):
        return self.module(x)


class DownsamplingBlock(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cond0_channels: Optional[int],
                 cond1_channels: Optional[int],
                 num_res_blocks: int,
                 dropout_prob: float,
                 use_attention: bool,
                 perform_downsampling: bool,
                 resample_with_res_block: bool,
                 use_conv_to_resample: bool,
                 attention_block_args: AttentionBlockArgs,
                 condition_bias: float = 1.0):
        super().__init__()
        self.use_attention = use_attention
        self.res_blocks = ModuleList()
        self.attention_blocks = ModuleList()
        self.perform_downsampling = perform_downsampling
        self.output_channels = []
        for j in range(num_res_blocks):
            self.res_blocks.append(ResBlock(
                in_channels=in_channels if j == 0 else out_channels,
                out_channels=out_channels,
                cond0_channels=cond0_channels,
                cond1_channels=cond1_channels,
                dropout_prob=dropout_prob,
                condition_bias=condition_bias))
            if use_attention:
                self.attention_blocks.append(AttentionBlock(out_channels, attention_block_args))
            self.output_channels.append(out_channels)
        if perform_downsampling:
            if resample_with_res_block:
                self.downsample = ResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    cond0_channels=cond0_channels,
                    cond1_channels=cond1_channels,
                    dropout_prob=dropout_prob,
                    sampling_mode=SamplingMode.DOWNSAMPING,
                    condition_bias=condition_bias)
            else:
                self.downsample = Arity3To1(Downsample(out_channels, use_conv_to_resample))
            self.output_channels.append(out_channels)

    def forward(self, h: Tensor, cond0: Optional[Tensor] = None, cond1: Optional[Tensor] = None) -> List[Tensor]:
        hs = []
        for i in range(len(self.res_blocks)):
            h = self.res_blocks[i].forward(h, cond0, cond1)
            if self.use_attention:
                h = self.attention_blocks[i].forward(h)
            hs.append(h)
        if self.perform_downsampling:
            hs.append(self.downsample(h, cond0, cond1))
        return hs


class UpsamplingBlock(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cond0_channels: Optional[int],
                 cond1_channels: Optional[int],
                 num_resnet_blocks: int,
                 skip_channels: List[int],
                 dropout_prob: float,
                 use_attention: bool,
                 perform_upsampling: bool,
                 resample_with_res_block: bool,
                 use_conv_to_resample: bool,
                 attention_block_args: AttentionBlockArgs,
                 condition_bias: float = 1.0):
        super().__init__()
        self.use_attention = use_attention
        self.resnet_blocks = ModuleList()
        self.attention_blocks = ModuleList()
        self.perform_upsampling = perform_upsampling
        for i in range(num_resnet_blocks):
            self.resnet_blocks.append(ResBlock(
                in_channels=(in_channels if i == 0 else out_channels) + skip_channels[i],
                out_channels=out_channels,
                cond0_channels=cond0_channels,
                cond1_channels=cond1_channels,
                dropout_prob=dropout_prob,
                condition_bias=condition_bias))
            if use_attention:
                self.attention_blocks.append(AttentionBlock(out_channels, attention_block_args))
        if perform_upsampling:
            if resample_with_res_block:
                self.upsample = ResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    cond0_channels=cond0_channels,
                    cond1_channels=cond1_channels,
                    sampling_mode=SamplingMode.UPSAMPLING,
                    dropout_prob=dropout_prob,
                    condition_bias=condition_bias)
            else:
                self.upsample = Arity3To1(Upsample(out_channels, use_conv_to_resample))

    def forward(self,
                h: Tensor,
                skips: List[Tensor],
                cond0: Optional[Tensor] = None,
                cond1: Optional[Tensor] = None) -> Tensor:
        for i in range(len(self.resnet_blocks)):
            h = self.resnet_blocks[i].forward(torch.concat([h, skips[i]], dim=1), cond0, cond1)
            if self.use_attention:
                h = self.attention_blocks[i].forward(h)
        if self.perform_upsampling:
            h = self.upsample.forward(h, cond0, cond1)
        return h


def compute_timestep_embedding(t: Tensor, out_channels: int):
    assert len(t.shape) == 2
    b, c = t.shape
    assert c == 1
    half_channels = out_channels // 2
    scale = -math.log(10000.0) / (half_channels - 1)
    log_times = scale * torch.arange(0, half_channels, device=t.device)
    times = torch.exp(log_times).reshape(1, half_channels) * t
    t_emb = torch.cat([torch.cos(times), torch.sin(times)], dim=1)
    if out_channels % 2 == 1:
        t_emb = torch.nn.functional.pad(t_emb, (1, 1), mode='constant')
    return t_emb


class TimeEmbedding(Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, t: Tensor):
        return compute_timestep_embedding(t, self.out_channels)


class UnetArgs:
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 model_channels: int = 64,
                 level_channel_multipliers: Optional[List[int]] = None,
                 level_use_attention: Optional[List[bool]] = None,
                 num_res_blocks_per_level: int = 2,
                 num_middle_res_blocks: int = 2,
                 time_embedding_channels: Optional[int] = None,
                 cond_input_channels: int = 4,
                 cond_internal_channels: int = 512,
                 attention_block_args: Optional[AttentionBlockArgs] = None,
                 dropout_prob: float = 0.1,
                 resample_with_res_block: bool = True,
                 use_conv_to_resample=False,
                 condition_bias: float = 1.0):
        assert len(level_channel_multipliers) == len(level_use_attention)
        assert not use_conv_to_resample or not resample_with_res_block

        if time_embedding_channels is None:
            time_embedding_channels = model_channels
        if level_channel_multipliers is None:
            level_channel_multipliers = [1, 2, 4, 8]
        if level_use_attention is None:
            level_use_attention = [False for _ in level_channel_multipliers]
        if attention_block_args is None:
            attention_block_args = AttentionBlockArgs(
                num_heads=1,
                num_head_channels=None,
                use_new_attention_order=False)

        self.condition_bias = condition_bias
        self.use_conv_to_resample = use_conv_to_resample
        self.resample_with_res_block = resample_with_res_block
        self.cond_internal_channels = cond_internal_channels
        self.dropout_prob = dropout_prob
        self.attention_block_args = attention_block_args
        self.time_embedding_channels = time_embedding_channels
        self.num_res_blocks_per_level = num_res_blocks_per_level
        self.level_use_attention = level_use_attention
        self.level_channel_multipliers = level_channel_multipliers
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_levels = len(level_channel_multipliers)
        self.num_middle_res_blocks = num_middle_res_blocks
        self.cond_input_channels = cond_input_channels


class Unet(Module):
    def __init__(self, args: UnetArgs):
        super().__init__()
        self.args = args

        self.time_embed = Sequential(
            TimeEmbedding(self.args.time_embedding_channels),
            Linear(self.args.time_embedding_channels, self.args.cond_internal_channels),
            SiLU(),
            Linear(self.args.cond_internal_channels, self.args.cond_internal_channels))

        self.cond_embed = Sequential(
            Linear(self.args.cond_input_channels, self.args.cond_internal_channels),
            SiLU(),
            Linear(self.args.cond_internal_channels, self.args.cond_internal_channels))

        self.first_conv = Conv2d(args.in_channels, args.model_channels, kernel_size=3, stride=1, padding=1)
        current_channels = args.model_channels
        channels = [current_channels]

        # Downsampling blocks
        self.down_blocks = ModuleList()
        for i in range(args.num_levels):
            out_channels = args.model_channels * args.level_channel_multipliers[i]
            perform_downsampling = i < args.num_levels - 1
            down_block = DownsamplingBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                cond0_channels=args.cond_internal_channels,
                cond1_channels=args.cond_internal_channels,
                num_res_blocks=args.num_res_blocks_per_level,
                dropout_prob=args.dropout_prob,
                use_attention=args.level_use_attention[i],
                perform_downsampling=perform_downsampling,
                attention_block_args=args.attention_block_args,
                resample_with_res_block=args.resample_with_res_block,
                use_conv_to_resample=args.use_conv_to_resample,
                condition_bias=args.condition_bias)
            self.down_blocks.append(down_block)
            current_channels = out_channels
            channels += down_block.output_channels

        # Middle blocks
        self.middle_blocks = ModuleList()
        for i in range(self.args.num_middle_res_blocks - 1):
            self.middle_blocks.append(ResBlock(
                in_channels=current_channels,
                out_channels=current_channels,
                cond0_channels=args.cond_internal_channels,
                cond1_channels=args.cond_internal_channels,
                dropout_prob=args.dropout_prob,
                condition_bias=args.condition_bias))
            self.middle_blocks.append(
                Arity3To1(AttentionBlock(num_channels=current_channels, args=args.attention_block_args)))
        self.middle_blocks.append(ResBlock(
            in_channels=current_channels,
            out_channels=current_channels,
            cond0_channels=args.cond_internal_channels,
            cond1_channels=args.cond_internal_channels,
            dropout_prob=args.dropout_prob,
            condition_bias=args.condition_bias))

        # Upsampling blocks
        self.up_blocks = ModuleList()
        for i in reversed(range(args.num_levels)):
            skip_channels = []
            for j in range(args.num_res_blocks_per_level + 1):
                skip_channels.append(channels.pop())
            perform_upsampling = i > 0
            out_channels = args.model_channels * args.level_channel_multipliers[i]
            up_block = UpsamplingBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                cond0_channels=args.cond_internal_channels,
                cond1_channels=args.cond_internal_channels,
                num_resnet_blocks=args.num_res_blocks_per_level + 1,
                skip_channels=skip_channels,
                dropout_prob=args.dropout_prob,
                use_attention=args.level_use_attention[i],
                perform_upsampling=perform_upsampling,
                attention_block_args=args.attention_block_args,
                resample_with_res_block=args.resample_with_res_block,
                use_conv_to_resample=args.use_conv_to_resample,
                condition_bias=args.condition_bias)
            self.up_blocks.append(up_block)
            current_channels = out_channels
        assert len(channels) == 0

        self.last = Sequential(
            GroupNorm32(current_channels),
            SiLU(),
            init_to_zero(Conv2d(current_channels, args.out_channels, kernel_size=3, stride=1, padding=1)))

    def forward(self, x: Tensor, t: Tensor, cond: Tensor):
        t_emb = self.time_embed(t)
        cond_emb = self.cond_embed(cond)
        hs = [self.first_conv(x)]
        for block in self.down_blocks:
            hs += block.forward(hs[-1], t_emb, cond_emb)
        h = hs[-1]
        for block in self.middle_blocks:
            h = block(h, t_emb, cond_emb)
        for block in self.up_blocks:
            skips = []
            for i in range(self.args.num_res_blocks_per_level + 1):
                skips.append(hs.pop())
            h = block.forward(h, skips, t_emb, cond_emb)
        assert len(hs) == 0
        return self.last(h)


class UnetWithFirstConvAddition(Module):
    def __init__(self, args: UnetArgs):
        super().__init__()
        self.args = args

        self.time_embed = Sequential(
            TimeEmbedding(self.args.time_embedding_channels),
            Linear(self.args.time_embedding_channels, self.args.cond_internal_channels),
            SiLU(),
            Linear(self.args.cond_internal_channels, self.args.cond_internal_channels))

        self.cond_embed = Sequential(
            Linear(self.args.cond_input_channels, self.args.cond_internal_channels),
            SiLU(),
            Linear(self.args.cond_internal_channels, self.args.cond_internal_channels))

        self.first_conv = Conv2d(args.in_channels, args.model_channels, kernel_size=3, stride=1, padding=1)
        current_channels = args.model_channels
        channels = [current_channels]

        # Downsampling blocks
        self.down_blocks = ModuleList()
        for i in range(args.num_levels):
            out_channels = args.model_channels * args.level_channel_multipliers[i]
            perform_downsampling = i < args.num_levels - 1
            down_block = DownsamplingBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                cond0_channels=args.cond_internal_channels,
                cond1_channels=args.cond_internal_channels,
                num_res_blocks=args.num_res_blocks_per_level,
                dropout_prob=args.dropout_prob,
                use_attention=args.level_use_attention[i],
                perform_downsampling=perform_downsampling,
                attention_block_args=args.attention_block_args,
                resample_with_res_block=args.resample_with_res_block,
                use_conv_to_resample=args.use_conv_to_resample,
                condition_bias=args.condition_bias)
            self.down_blocks.append(down_block)
            current_channels = out_channels
            channels += down_block.output_channels

        # Middle blocks
        self.middle_blocks = ModuleList()
        for i in range(self.args.num_middle_res_blocks - 1):
            self.middle_blocks.append(ResBlock(
                in_channels=current_channels,
                out_channels=current_channels,
                cond0_channels=args.cond_internal_channels,
                cond1_channels=args.cond_internal_channels,
                dropout_prob=args.dropout_prob,
                condition_bias=args.condition_bias))
            self.middle_blocks.append(
                Arity3To1(AttentionBlock(num_channels=current_channels, args=args.attention_block_args)))
        self.middle_blocks.append(ResBlock(
            in_channels=current_channels,
            out_channels=current_channels,
            cond0_channels=args.cond_internal_channels,
            cond1_channels=args.cond_internal_channels,
            dropout_prob=args.dropout_prob,
            condition_bias=args.condition_bias))

        # Upsampling blocks
        self.up_blocks = ModuleList()
        for i in reversed(range(args.num_levels)):
            skip_channels = []
            for j in range(args.num_res_blocks_per_level + 1):
                skip_channels.append(channels.pop())
            perform_upsampling = i > 0
            out_channels = args.model_channels * args.level_channel_multipliers[i]
            up_block = UpsamplingBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                cond0_channels=args.cond_internal_channels,
                cond1_channels=args.cond_internal_channels,
                num_resnet_blocks=args.num_res_blocks_per_level + 1,
                skip_channels=skip_channels,
                dropout_prob=args.dropout_prob,
                use_attention=args.level_use_attention[i],
                perform_upsampling=perform_upsampling,
                attention_block_args=args.attention_block_args,
                resample_with_res_block=args.resample_with_res_block,
                use_conv_to_resample=args.use_conv_to_resample,
                condition_bias=args.condition_bias)
            self.up_blocks.append(up_block)
            current_channels = out_channels
        assert len(channels) == 0

        self.last = Sequential(
            GroupNorm32(current_channels),
            SiLU(),
            init_to_zero(Conv2d(current_channels, args.out_channels, kernel_size=3, stride=1, padding=1)))

    def forward(self, x: Tensor, t: Tensor, cond: Tensor, first_conv_addition: Tensor):
        t_emb = self.time_embed(t)
        cond_emb = self.cond_embed(cond)
        first_conv = self.first_conv(x)
        hs = [first_conv + first_conv_addition]
        for block in self.down_blocks:
            hs += block.forward(hs[-1], t_emb, cond_emb)
        h = hs[-1]
        for block in self.middle_blocks:
            h = block(h, t_emb, cond_emb)
        for block in self.up_blocks:
            skips = []
            for i in range(self.args.num_res_blocks_per_level + 1):
                skips.append(hs.pop())
            h = block.forward(h, skips, t_emb, cond_emb)
        assert len(hs) == 0
        return self.last(h)
