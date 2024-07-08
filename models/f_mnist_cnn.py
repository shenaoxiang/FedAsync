import torch.nn as nn
import torch.nn.functional as F

class FashionMNIST_CNN1(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN1, self).__init__()
        # 第一个卷积层，输入通道为1，输出通道为16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层，输入通道为16，输出通道为32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层，输入特征为32*7*7，输出特征为128
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # 输出层，输出类别数为10
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FashionMNIST_CNN2(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN2, self).__init__()
        # 第一个卷积层，输入通道为1，输出通道为16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization层
        # 第一个池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层，输入通道为16，输出通道为32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization层
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层，输入特征为32*7*7（经过两次池化），输出特征为10（对应类别数）
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x


class FashionMNIST_CNN(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()
        # 第一个卷积层，输入通道为1，输出通道为16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 第一个池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层，输入通道为16，输出通道为32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层，输入特征为32*7*7（经过两次池化），输出特征为10（对应类别数）
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x


class FashionMNIST_CNN11(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN11, self).__init__()
        # 第一个卷积层，输入通道为1，输出通道为16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积层，输入通道为16，输出通道为32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层，输入特征为32*7*7，输出特征为128
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # 输出层，输出类别数为10
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
