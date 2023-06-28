import torch.nn as nn
import torch.nn.functional as f


class Residual(nn.Module):
    """ 残差块 """
    def __init__(self, in_channels, num_channels, strides=1, use_1x1_conv=False):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1_conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, input_x):
        output_y = f.relu(self.bn1(self.conv1(input_x)))
        output_y = self.bn2(self.conv2(output_y))

        if self.conv3:
            input_x = self.conv3(input_x)

        return f.relu(output_y + input_x)

def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    """ 残差模块 """
    blk = []

    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, num_channels, strides=2, use_1x1_conv=True))
        else:
            blk.append(Residual(num_channels, num_channels))

    return blk

""" ResNet-18搭建 """
def create_model():
    b1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
    )

    # 残差模块一 Residual(64, 64), Residual(64, 64)
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))

    # 残差模块二 Residual(64, 128), Residual(128, 128)
    b3 = nn.Sequential(*resnet_block(64, 128, 2))

    # 残差模块三 Residual(128, 256), Residual(256, 256)
    b4 = nn.Sequential(*resnet_block(128, 256, 2))

    # 残差模块四 Residual(256, 512), Residual(512, 512)
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    layers = [b1, b2, b3, b4, b5,
              nn.AdaptiveAvgPool2d((1, 1)),
              nn.Flatten(),
              nn.Linear(512, 100, bias=True)]

    return nn.Sequential(*layers)