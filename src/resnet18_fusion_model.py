import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Fusion(nn.Module):
    """基于ResNet18的RGB-Depth融合模型"""
    def __init__(self, dropout_rate=0.5):
        super().__init__()

        # RGB分支：ResNet18
        self.rgb_net = models.resnet18(weights=None)
        num_ftrs_rgb = self.rgb_net.fc.in_features  # 512
        self.rgb_net.fc = nn.Identity()

        # 深度分支：ResNet18（修改第一层）
        self.depth_net = models.resnet18(weights=None)
        self.depth_net.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_ftrs_depth = self.depth_net.fc.in_features  # 512
        self.depth_net.fc = nn.Identity()

        # 融合回归头
        self.regressor = nn.Sequential(
            nn.Linear(num_ftrs_rgb + num_ftrs_depth, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 1)
        )

        # 初始化深度分支第一层
        nn.init.kaiming_normal_(
            self.depth_net.conv1.weight,
            mode='fan_out',
            nonlinearity='relu'
        )

    def forward(self, rgb, depth):
        rgb_features = self.rgb_net(rgb)
        depth_features = self.depth_net(depth)
        fused = torch.cat([rgb_features, depth_features], dim=1)
        return self.regressor(fused)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
