from torch import nn
from efficientCV.layers.convolutions import ConvBnAct, DepthwiseSepConv


class MobileNetV1(nn.Module):
    """Implementation of MobileNet V1"""

    def __init__(self, img_channel=3, num_classes=1000):
        super(MobileNetV1, self).__init__()

        last_channel = 1024
        self.stages = self.build_stage(img_channel)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(in_features=last_channel, out_features=num_classes)

    def build_stage(self, img_channel):
        """Building mobilenet stages.
        Feature extraction module for mobilenet.
        """

        layer_configs = [
            [32],
            [64],
            [128, 128],
            [256, 256],
            [512, 512, 512, 512, 512, 512],
            [1024, 1024],
        ]

        init_channels = layer_configs[0][0]
        stages = [
            ConvBnAct(
                img_channel,
                init_channels,
                kernel_size=3,
                stride=2,
                activation=nn.ReLU(inplace=True),
            )
        ]

        in_channels = init_channels
        for i, channels in enumerate(layer_configs[1:]):
            for j, out_channels in enumerate(channels):

                stride = 2 if (j == 0) and (i != 0) else 1
                stages.append(
                    DepthwiseSepConv(
                        in_channels, out_channels, kernel_size=3, stride=stride
                    )
                )

                in_channels = out_channels
        return nn.Sequential(*stages)

    def forward(self, x):
        x = self.stages(x)
        x = self.avg_pool(x)
        x = self.linear(x.view(x.shape[0], -1))
        return x
