import torch.nn as nn


class CifarCNN(nn.Module):
    """修改后的CNN模型，包含批量归一化层和Dropout层。"""
    def __init__(self):
        """CNN构造函数。"""
        super(CifarCNN, self).__init__()
        # 卷积层
        self.conv_layer = nn.Sequential(
            # 第1个卷积层
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 第2个卷积层
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 第3个卷积层
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Dropout层
            nn.Dropout2d(p=0.05)
        )
        # 全连接层
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),  # 输入图像尺寸为 32x32
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),  # 在全连接层之间添加Dropout
            nn.Linear(512, 10)  # 有10个输出类别
        )
    
    def forward(self, x):
        """定义模型的前向传播路径。"""
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # 展平卷积层输出
        x = self.fc_layer(x)
        return x