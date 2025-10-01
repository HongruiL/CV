import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFusion(nn.Module):
    """基于ResNet18的RGB-Depth融合模型"""
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        # RGB分支：标准ResNet18
        self.rgb_net = models.resnet18(weights=None)
        num_ftrs_rgb = self.rgb_net.fc.in_features  # 512
        self.rgb_net.fc = nn.Identity()  # 移除分类层
        
        # 深度分支：修改第一层接受单通道输入
        self.depth_net = models.resnet18(weights=None)
        self.depth_net.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_ftrs_depth = self.depth_net.fc.in_features  # 512
        self.depth_net.fc = nn.Identity()
        
        # 融合后的回归头
        self.regressor = nn.Sequential(
            nn.Linear(num_ftrs_rgb + num_ftrs_depth, 512),  # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 1)
        )
        
        # 初始化深度分支的第一个卷积层
        # 因为我们改变了输入通道数，需要重新初始化
        nn.init.kaiming_normal_(
            self.depth_net.conv1.weight, 
            mode='fan_out', 
            nonlinearity='relu'
        )
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: (B, 3, 224, 224)
            depth: (B, 1, 224, 224)
        Returns:
            (B, 1) 卡路里预测
        """
        rgb_features = self.rgb_net(rgb)      # (B, 512)
        depth_features = self.depth_net(depth)  # (B, 512)
        
        # 拼接特征
        fused = torch.cat([rgb_features, depth_features], dim=1)  # (B, 1024)
        
        # 回归预测
        output = self.regressor(fused)  # (B, 1)
        
        return output


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试代码
if __name__ == '__main__':
    model = ResNetFusion(dropout_rate=0.5)
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224)
    depth = torch.randn(batch_size, 1, 224, 224)
    
    output = model(rgb, depth)
    print(f"输入 - RGB: {rgb.shape}, Depth: {depth.shape}")
    print(f"输出: {output.shape}")
    print(f"输出值: {output.squeeze()}")