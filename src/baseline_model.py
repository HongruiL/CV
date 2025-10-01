import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    Baseline CNN 用于卡路里预测
    
    架构：5个卷积块 + 2个全连接层
    输入：(B, 3, 224, 224) RGB图像
    输出：(B, 1) 卡路里预测值
    """
    def __init__(self, dropout_rate=0.5):
        super(BaselineCNN, self).__init__()
        
        # 卷积块 1: 3 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 224 -> 112
        )
        
        # 卷积块 2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )
        
        # 卷积块 3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )
        
        # 卷积块 4: 128 -> 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 28 -> 14
        )
        
        # 卷积块 5: 256 -> 512
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 14 -> 7
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)  # 输出1个值
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, 3, 224, 224) RGB图像
        
        Returns:
            (B, 1) 卡路里预测值
        """
        x = self.conv1(x)   # (B, 32, 112, 112)
        x = self.conv2(x)   # (B, 64, 56, 56)
        x = self.conv3(x)   # (B, 128, 28, 28)
        x = self.conv4(x)   # (B, 256, 14, 14)
        x = self.conv5(x)   # (B, 512, 7, 7)
        
        # 全局平均池化
        x = self.global_avg_pool(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 512)
        
        # 全连接层
        x = self.fc(x)  # (B, 1)
        
        # 确保输出非负（卡路里不能为负）
        x = F.relu(x)
        
        return x


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试代码
if __name__ == '__main__':
    # 创建模型
    model = BaselineCNN(dropout_rate=0.5)
    
    # 统计参数
    num_params = count_parameters(model)
    print(f"模型参数量: {num_params:,}")
    
    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\n输入shape: {dummy_input.shape}")
    
    # 前向传播
    output = model(dummy_input)
    print(f"输出shape: {output.shape}")
    print(f"输出值: {output.squeeze()}")
    
    # 检查梯度流
    print(f"\n模型可训练: {model.training}")
    print(f"第一层权重requires_grad: {model.conv1[0].weight.requires_grad}")