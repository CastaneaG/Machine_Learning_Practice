import torch
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    """
    for cifar-10 dataset.
    """

    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32]
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5, stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),

        )
        # flatten 打平
        # fc
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

        # tmp = torch.randn(2,3,32,32)
        # out = self.conv_unit(tmp)
        # print('conv out:',out.shape)

        # nn.CrossEntropyLoss() 之前包含Softmax 所以不需要自己在进行Softmax
        # self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        """

        :param x: [b, 3, 32, 32]
        :return: logits [b, 10]
        """

        # x.size(0)等同于x.shape[0]
        batch_size = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batch_size,16*5*5)
        logits = self.fc_unit(x)

        return logits

        # nn.CrossEntropyLoss() 之前包含Softmax 所以不需要自己在进行Softmax
        # [b, 10] 在维度为1 的 10 这个维度做交叉熵
        # pred = F.softmax(logits, dim = 1)


def main():
    net = Lenet5()
    tmp = torch.randn(2,3,32,32)
    out = net(tmp)
    print('lenet out:',out.shape)

if __name__ == '__main__':
    main()