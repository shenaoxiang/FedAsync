import torch.nn as nn
import torch.nn.functional as F


class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 二维Dropout层，用于减少过拟合，提高模型的泛化能力。丢弃率为5%
        # 仅在训练时使用 Dropout，而在测试（或验证）时不使用，默认丢弃率为0.5
        self.conv2_drop = nn.Dropout2d(p=0.05)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # 用于改变张量的形状而不改变其数据。它返回一个新的张量，具有指定的形状
        # [N, C, H, W] -> 其中N是批量大小，C是通道数，H是高度，W是宽度
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x