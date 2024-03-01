from torch import nn
from efficientCV.layers.convolutions import ConvBnAct, InvertedResidual
from efficientCV.utils.utils import make_divisible


class MobileNetV2(nn.Module):

    def __init__(self, img_channel=3, n_class=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()

        last_channel = 1280
        self.last_channel = (
            make_divisible(last_channel * width_mult)
            if width_mult > 1.0
            else last_channel
        )
        self.stages = self.build_stage(img_channel, self.last_channel)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        # building classifier
        self.linear = nn.Linear(self.last_channel, n_class)

    def build_stage(self, img_channel, last_channel, width_mult=1.0):
        # expand ratio , channels , no of layers (repeat), stride
        model_config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        init_channels = 32
        last_channel = last_channel
        layers = [ConvBnAct(3, init_channels, kernel_size=3, stride=2, activation=None)]
        in_channel = init_channels
        for t, c, n, s in model_config:
            out_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    layers += [
                        InvertedResidual(in_channel, out_channel, s, expand_factor=t)
                    ]
                else:
                    layers += [
                        InvertedResidual(in_channel, out_channel, 1, expand_factor=t)
                    ]
                in_channel = out_channel

        layers += [
            ConvBnAct(
                in_channel, last_channel, kernel_size=1, stride=1, activation=None
            )
        ]

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stages(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return
