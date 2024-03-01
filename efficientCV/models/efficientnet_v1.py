from typing import Tuple
from torch import nn
import torch
from efficientCV.layers.convolutions import ConvBnAct, MBConvBlock
from efficientCV.utils.utils import _round_filters, _round_repeats


class EfficientNetV1(nn.Module):
    """Implementation of Efficientnet V1 ."""

    def __init__(
        self,
        model_configs: Tuple[float, float, int, float],
        num_classes: int = 1000,
        device: str = "cpu",
    ):
        super(EfficientNetV1, self).__init__()
        self.device = device
        self.last_channel = 1280
        self.stages = self.build_stage(model_configs)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=self.last_channel, out_features=num_classes)

    def build_stage(
        self, model_configs: Tuple[float, float, int, float]
    ) -> nn.Sequential:
        """Building efficientnet stages.
        Feature extraction module for EfficientNet
        """

        width_mult = model_configs[0]
        depth_mult = model_configs[1]
        drop_connect_rate = model_configs[3]

        # expantion_factor,channels, no_layers(repeat), stride, kernel_size
        layer_config = [
            [1, 16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6, 24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6, 40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6, 80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->
            [6, 320, 1, 1, 3],  # MBConv6_3x3, SE,   7 ->   7
        ]

        out_channels = _round_filters(32, width_mult)

        stages = [
            ConvBnAct(3, out_channels, 3, stride=2, activation=nn.SiLU(inplace=True))
        ]
        in_channels = out_channels

        total_stages = sum([i[2] for i in layer_config])
        block_id = 0

        for t, c, n, s, k in layer_config:

            layers = []
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):

                stride = s if i == 0 else 1
                drop_connect_prob = drop_connect_rate * float(block_id) / total_stages

                layers.append(
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        t,
                        drop_connect_prob,
                        stride=stride,
                        kernel_size=k,
                        device=self.device,
                    )
                )

                in_channels = out_channels
                block_id += 1
            stages.append(nn.Sequential(*layers))
        stages += [
            ConvBnAct(
                in_channels, self.last_channel, 1, 1, activation=nn.SiLU(inplace=True)
            )
        ]

        return nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(x)
        x = self.avg_pool(x)
        x = self.linear(x.view(x.shape[0], -1))
        return x
