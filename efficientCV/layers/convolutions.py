import torch
from torch import nn
from efficientCV.utils.ops import DropConnect


class ConvBnAct(nn.Module):

    "Combination of convolution , batch-norm and activations functions ."

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation,
        groups: int = 1,
        bias: bool = False,
    ):

        super(ConvBnAct, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            activation if activation else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    """

    def __init__(self, in_channels: int, reduced_channels: int):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.se(x)
        return x * y


class DepthwiseSepConv(nn.Module):
    "Depth-wise seperable convolution block ."

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super(DepthwiseSepConv, self).__init__()

        self.depth_wise = ConvBnAct(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            activation=nn.ReLU(inplace=True),
        )

        self.point_wise = ConvBnAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            activation=nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


class InvertedResidual(nn.Module):

    "Inverted residual block."

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, expand_factor: int
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        print(f"expand_factor {type(expand_factor)}")
        hidden_dim = int(in_channels * expand_factor)
        self.skip_connection = (self.stride == 1) and (in_channels == out_channels)

        if expand_factor == 1:
            self.conv = nn.Sequential(
                ConvBnAct(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_dim,
                    activation=nn.ReLU(inplace=True),
                ),  # depth wise convolution
                ConvBnAct(
                    hidden_dim, out_channels, kernel_size=1, stride=1, activation=None
                ),  # point wise linear
            )
        else:
            self.conv = nn.Sequential(
                ConvBnAct(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    activation=nn.ReLU(inplace=True),
                ),  # point wise
                ConvBnAct(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_dim,
                    activation=nn.ReLU(inplace=True),
                ),  # depth wise convolution
                ConvBnAct(
                    hidden_dim, out_channels, kernel_size=1, stride=1, activation=None
                ),  # point wise linear
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MBConvBlock(nn.Module):

    "Mobile inverted-resiudual block"

    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        expansion_factor: int,
        drop_connect_prob: float,
        kernel_size: int,
        stride: int,
        reduction_factor: int = 4,
        device: str = "cpu",
    ):

        super(MBConvBlock, self).__init__()
        self.device = device

        self.drop_connect_prob = drop_connect_prob
        self.skip_connection = stride == 1 and input_channels == out_channels

        expand_channels = int(input_channels * expansion_factor)
        reduced_channels = max(1, int(input_channels // reduction_factor))

        self.expand = (
            nn.Identity()
            if (expansion_factor == 1)
            else (
                ConvBnAct(
                    input_channels,
                    expand_channels,
                    kernel_size=1,
                    stride=1,
                    activation=nn.SiLU(inplace=True),
                )
            )
        )

        self.depthwise_conv = ConvBnAct(
            expand_channels,
            expand_channels,
            kernel_size,
            stride,
            groups=expand_channels,
            activation=nn.SiLU(inplace=True),
        )

        self.se = SqueezeExcitationBlock(expand_channels, reduced_channels)
        self.pointwise_conv = ConvBnAct(
            expand_channels, out_channels, kernel_size=1, stride=1, activation=None
        )

        self.drop_connect = DropConnect(self.drop_connect_prob, self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.expand(x)
        result = self.depthwise_conv(result)
        result = self.se(result)
        result = self.pointwise_conv(result)

        if self.skip_connection:
            result = self.drop_connect(x)
            result += x
        return result


class FuseMvConvBlock(nn.Module):

    "Fused-Mobile inverted-resiudual block"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: int,
        drop_connect_prob: float,
        stride: int,
        kernel_size: int,
        reduction_factor: int = 4,
        device: str = "cpu",
    ):
        super(FuseMvConvBlock, self).__init__()
        self.device = device

        self.drop_connect_prob = drop_connect_prob
        self.skip_connection = stride == 1 and in_channels == out_channels

        # reduced_channels = max(1, int(in_channels // reduction_factor))
        expand_channels = int(in_channels * expansion_factor)

        self.fuse_conv = nn.Sequential(
            (
                ConvBnAct(
                    in_channels,
                    expand_channels,
                    kernel_size=3,
                    stride=stride,
                    activation=nn.ReLU(inplace=True),
                )
            ),
            ConvBnAct(
                expand_channels, out_channels, kernel_size=1, stride=1, activation=None
            ),
        )
        self.drop_connect = DropConnect(self.drop_connect_prob, self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.fuse_conv(x)

        if self.skip_connection:
            result = self.drop_connect(x)
            result += x
        return result
