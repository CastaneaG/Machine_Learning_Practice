import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1):
        """

        :param channel_in:
        :param channel_out:
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_out)

        self.extra = nn.Sequential()
        if channel_out != channel_in:
            # extra本身是空的，若输入输出维度不一样，extra以保证x与out可以想家
            self.extra = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channel_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        # 原始论文里没有第二个Relu
        out = F.relu(self.bn2(self.conv2(out)))

        # shortcut
        out = self.extra(x) + out

        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )

        self.block1 = ResBlock(64, 128, stride=2)
        self.block2 = ResBlock(128, 256, stride=2)
        self.block3 = ResBlock(256, 512, stride=2)
        self.block4 = ResBlock(512, 512, stride=2)
        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # 无论输入长和宽是多少 都会输出一个像素
        x = F.adaptive_avg_pool2d(x, [1, 1])

        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x
