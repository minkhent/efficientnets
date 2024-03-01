from torch import nn
from efficientCV.layers.convolutions import ConvBnAct, MBConvBlock, FuseMvConvBlock


class EfficientNetV2(nn.Module):
    """Implementation of Efficientnet V2 ."""

    def __init__(self, layer_config, num_classes=1000, device="cpu"):
        super(EfficientNetV2, self).__init__()

        self.device = device
        self.last_channel = 1280
        self.stages = self.build_stage(layer_config, drop_connect_rate=0.2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=self.last_channel, out_features=num_classes)

    def build_stage(self, layer_config, drop_connect_rate):
        """Building efficientnet stages.
        Feature extraction module for EfficientNet.
        """

        out_channels = 24

        total_stages = sum([n for layer, (t, k, s, c, n) in layer_config.items()])
        stages = [
            ConvBnAct(
                3,
                out_channels,
                kernel_size=3,
                stride=2,
                activation=nn.SiLU(inplace=True),
            )
        ]

        in_channels = out_channels
        block_id = 0

        for layer, (t, k, s, c, n) in layer_config.items():

            out_channels = c
            layers = []

            for i in range(n):
                stride = s if i == 0 else 1
                drop_connect_prob = (
                    drop_connect_rate * float(block_id) / total_stages
                )  # init drop connect probdepend on depth

                if layer.startswith("Fused"):
                    layers.append(
                        FuseMvConvBlock(
                            in_channels,
                            out_channels,
                            t,
                            drop_connect_prob,
                            stride=stride,
                            kernel_size=k,
                            device=self.device,
                        )
                    )
                else:
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

    def forward(self, x):
        x = self.stages(x)
        x = self.avg_pool(x)
        x = self.linear(x.view(x.shape[0], -1))
        return x


efficient_v2_s = {
    # expansion_factor, kernel_size, stride, ,channels, no_of layers
    "FusedMvConv_1": (1, 3, 1, 24, 2),
    "FusedMvConv_2": (4, 3, 2, 48, 4),
    "FusedMvConv_3": (4, 3, 2, 64, 4),
    "MvConv_1": (4, 3, 2, 128, 6),
    "MvConv_2": (6, 3, 1, 160, 9),
    "MvConv_3": (6, 3, 2, 256, 15),
}

efficient_v2_m = {
    "FusedMvConv_1": (1, 3, 1, 24, 3),
    "FusedMvConv_2": (4, 3, 2, 48, 5),
    "FusedMvConv_3": (4, 3, 2, 80, 5),
    "MvConv_1": (4, 3, 2, 160, 7),
    "MvConv_2": (6, 3, 1, 176, 14),
    "MvConv_3": (6, 3, 2, 304, 18),
    "MvConv_4": (6, 3, 1, 512, 5),
}

efficient_v2_l = {
    "FusedMvConv_1": (1, 3, 1, 32, 4),
    "FusedMvConv_2": (4, 3, 2, 64, 7),
    "FusedMvConv_3": (4, 3, 2, 96, 7),
    "MvConv_1": (4, 3, 2, 192, 10),
    "MvConv_2": (6, 3, 1, 224, 19),
    "MvConv_3": (6, 3, 2, 384, 25),
    "MvConv_4": (6, 3, 1, 640, 7),
}

efficient_v2_xl = {
    "FusedMvConv_1": (1, 3, 1, 32, 4),
    "FusedMvConv_2": (4, 3, 2, 64, 8),
    "FusedMvConv_3": (4, 3, 2, 96, 8),
    "MvConv_1": (4, 3, 2, 192, 16),
    "MvConv_2": (6, 3, 1, 256, 24),
    "MvConv_3": (6, 3, 2, 512, 32),
    "MvConv_4": (6, 3, 1, 640, 8),
}
