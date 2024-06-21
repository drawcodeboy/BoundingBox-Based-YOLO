import torch
from torch import nn

from torchsummary import summary

# CNN Architectrue -> DarkNet

architecture_config = [
    # kernel_size, filters, stride, padding
    (7, 64, 2, 3),
    # M is simply maspooling with stride=2 and kernel=2
    'M',
    (3, 192, 1, 1),
    'M',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'M',
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'M',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CBABlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CBABlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            # 첫 번째 x = (7,64,2,3)
            if type(x) == tuple:
                layers += [
                        CBABlock( # 사람이 정의한 block
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                    ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CBABlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CBABlock(
                            conv1[1],
                            conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        #In original paper this shuld be
        #nn.Linear(1024*S*S, 4096),
        #nn.LeakyReLU(0.1),
        #nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            # S, B, C = 7, 2, 20
            nn.Linear(496, S * S * (C + B * 5)),
        )

class CBABlock2(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CBABlock2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # 변경사항 (1): Batch Normalization -> Layer Normalization
        # https://discuss.pytorch.org/t/is-there-a-layer-normalization-for-conv2d/7595/4
        self.layernorm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        # 변경사항 (2): Leaky ReLU -> GELU
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.layernorm(self.conv(x)))

class Yolov1_tuned(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1_tuned, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    # 변경사항 (3): 가중치 초기화 Xaiver
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight.data)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            # 첫 번째 x = (7,64,2,3)
            if type(x) == tuple:
                layers += [
                        CBABlock2( # 사람이 정의한 block
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                    ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CBABlock2(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CBABlock2(
                            conv1[1],
                            conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        #In original paper this shuld be
        #nn.Linear(1024*S*S, 4096),
        #nn.LeakyReLU(0.1),
        #nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.GELU(),
            # S, B, C = 7, 2, 20
            nn.Linear(496, S * S * (C + B * 5)),
        )

if __name__ == '__main__':
    model1 = Yolov1(split_size=7, num_boxes=2, num_classes=3)
    summary(model1, (3, 448, 448))
    model2 = Yolov1_tuned(split_size=7, num_boxes=2, num_classes=3)
    summary(model2, (3, 448, 448))