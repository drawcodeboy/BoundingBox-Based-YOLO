import torch
from torch import nn

from torchsummary import summary

import sys

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
    
class MLPBlock(nn.Module):
    r"""
        헤당 MLP Block은 구현상으로 병렬적인 연산 처리 방식이 불가능하기 때문에
        사용하지 않는다. -> 1x1 Conv로 대체하였다. -> CoordConvBlock
    """
    def __init__(self, in_features=4):
        super(MLPBlock, self).__init__()
        
        self.l1 = nn.Linear(in_features, 2*in_features, bias=False)
        self.l2 = nn.Linear(2*in_features, 1, bias=False)
        self.acti = nn.GELU()
    
    def forward(self, x):
        return self.l2(self.acti(self.l1(x)))

class CoordConvBlock(nn.Module):
    def __init__(self, in_channels=4, groups=2):
        super(CoordConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(groups*in_channels, groups*2*in_channels, kernel_size=1, groups=groups, bias=False)
        self.conv2 = nn.Conv2d(groups*2*in_channels, groups, kernel_size=1, groups=groups, bias=False)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        return self.conv2(self.gelu(self.conv1(x)))

class BBBasedYolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(BBBasedYolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        self.S, self.B, self.C = (kwargs['split_size'], kwargs['num_boxes'], kwargs['num_classes'])
        
        # 변경 사항 (4): Objectness Score based Bounding Boxes : 병렬 연산 불가 -> 1x1 conv로 해결
        '''
        self.mlp_set = nn.ModuleList([
                            nn.ModuleList([
                                nn.ModuleList([MLPBlock(4) for _ in range(0, self.B)]) 
                            for __ in range(0, self.S)]) 
                        for ___ in range(0, self.S)])
        '''
        # 시그모이드의 경우 여기서는 Gradient Vanshing을 야기하지는 않는다.
        # self.sigmoid = nn.Sigmoid()
        
        # 셀 별 Bounding Box의 수만큼 생성
        self.coord_blk = CoordConvBlock(in_channels=4, groups=self.B)
        self.sigmoid = nn.Sigmoid()
        
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.darknet(x)
        x = self.fcs(torch.flatten(x, start_dim=1))
        x = x.reshape(x.shape[0], self.S, self.S, self.C + self.B*5)
        
        # Bounding Box 좌표 추출 -> Concat (N, (셀 별 박스 수 * 좌표 수), self.S, self.S)
        confidence_bbox = torch.cat((x[..., self.C+1:self.C+5].reshape(-1, 4, self.S, self.S).squeeze(1), 
                                       x[..., self.C+6:self.C+10].reshape(-1, 4, self.S, self.S).squeeze(1)), dim=1)
        
        confidence_bbox_score = self.sigmoid(self.coord_blk(confidence_bbox)).reshape(-1, self.S, self.S, self.B)
        
        x[..., self.C] *= confidence_bbox_score[..., 0]
        x[..., self.C+5] *= confidence_bbox_score[..., 1]
        
        # 변경 사항 4에 대한 forward: 병렬 처리 불가 -> 1x1 conv로 해결
        '''
        for i in range(0, self.S):
            for j in range(0, self.S):
                    x[..., i, j, self.C] = x[..., i, j, self.C] * self.sigmoid(self.mlp_set[i][j][0](x[..., i, j, (self.C+1):(self.C+5)])).reshape(x.shape[0])
                    x[..., i, j, (self.C+5)] = x[..., i, j, (self.C+5)] * self.sigmoid(self.mlp_set[i][j][1](x[..., i, j, (self.C+6):(self.C+10)])).reshape(x.shape[0])
        '''
        
        x = torch.flatten(x, start_dim=1)
        return x
    
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
                        CBABlock2(
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
            nn.Linear(122880, 496),
            nn.Dropout(0.0),
            nn.GELU(),
            # S, B, C = 7, 2, 20
            nn.Linear(496, S * S * (C + B * 5)),
        )

if __name__ == '__main__':
    # temp = CoordConvBlock()
    #summary(temp, (8, 7, 7))
    #sys.exit()
    
    model = BBBasedYolov1(split_size=7, num_boxes=2, num_classes=3)
    summary(model, (3, 540, 960))
    x = torch.randn(1, 3, 540, 960)
    o = model(x)
    print(o.shape)