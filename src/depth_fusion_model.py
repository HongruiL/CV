import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """跨模态注意力融合"""
    def __init__(self, rgb_dim=512, depth_dim=256):
        super().__init__()
        
        # 通道注意力
        self.rgb_channel_att = nn.Sequential(
            nn.Linear(rgb_dim, rgb_dim // 16),
            nn.ReLU(),
            nn.Linear(rgb_dim // 16, rgb_dim),
            nn.Sigmoid()
        )
        
        self.depth_channel_att = nn.Sequential(
            nn.Linear(depth_dim, depth_dim // 16),
            nn.ReLU(),
            nn.Linear(depth_dim // 16, depth_dim),
            nn.Sigmoid()
        )
        
        # 跨模态权重
        self.modal_weight = nn.Sequential(
            nn.Linear(rgb_dim + depth_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, rgb_feat, depth_feat):
        # 通道注意力
        rgb_att = self.rgb_channel_att(rgb_feat)
        depth_att = self.depth_channel_att(depth_feat)
        
        rgb_enhanced = rgb_feat * rgb_att
        depth_enhanced = depth_feat * depth_att
        
        # 计算模态权重
        combined = torch.cat([rgb_enhanced, depth_enhanced], dim=1)
        weights = self.modal_weight(combined)
        
        # 加权融合
        rgb_weighted = rgb_enhanced * weights[:, 0:1]
        depth_weighted = depth_enhanced * weights[:, 1:2]
        
        return torch.cat([rgb_weighted, depth_weighted], dim=1)

class DepthFusionCNN(nn.Module):
    """
    双流深度融合网络
    
    架构：
    - RGB分支：提取外观特征
    - 深度分支：提取体积特征
    - 融合层：结合两种特征
    """
    def __init__(self, fusion_method='concat', dropout_rate=0.5):
        """
        Args:
            fusion_method: 'concat', 'add', 'attention'
            dropout_rate: Dropout比例
        """
        super(DepthFusionCNN, self).__init__()
        self.fusion_method = fusion_method
        
        # ========== RGB分支 ==========
        self.rgb_conv1 = self._make_conv_block(3, 32)
        self.rgb_conv2 = self._make_conv_block(32, 64)
        self.rgb_conv3 = self._make_conv_block(64, 128)
        self.rgb_conv4 = self._make_conv_block(128, 256)
        self.rgb_conv5 = self._make_conv_block(256, 512)
        
        # ========== 深度分支（稍小容量） ==========
        self.depth_conv1 = self._make_conv_block(1, 16)
        self.depth_conv2 = self._make_conv_block(16, 32)
        self.depth_conv3 = self._make_conv_block(32, 64)
        self.depth_conv4 = self._make_conv_block(64, 128)
        self.depth_conv5 = self._make_conv_block(128, 256)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ========== 融合层 ==========
        if fusion_method == 'concat':
            fusion_dim = 512 + 256
            
        elif fusion_method == 'add':
            # 投影到相同维度
            self.depth_projection = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512)
            )
            fusion_dim = 512
            
        elif fusion_method == 'attention':
            # 使用改进的注意力
            self.cross_attention = CrossModalAttention(512, 256)
            fusion_dim = 512 + 256

        # ========== 改进的回归头 ==========
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels, out_channels):
        """构建卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def _initialize_weights(self):
        """权重初始化"""
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
    
    def forward(self, rgb, depth):
        """
        前向传播
        
        Args:
            rgb: (B, 3, 224, 224)
            depth: (B, 1, 224, 224)
        
        Returns:
            (B, 1) 卡路里预测
        """
        # RGB分支
        rgb_feat = self.rgb_conv1(rgb)
        rgb_feat = self.rgb_conv2(rgb_feat)
        rgb_feat = self.rgb_conv3(rgb_feat)
        rgb_feat = self.rgb_conv4(rgb_feat)
        rgb_feat = self.rgb_conv5(rgb_feat)
        rgb_feat = self.global_pool(rgb_feat).view(rgb_feat.size(0), -1)
        
        # 深度分支
        depth_feat = self.depth_conv1(depth)
        depth_feat = self.depth_conv2(depth_feat)
        depth_feat = self.depth_conv3(depth_feat)
        depth_feat = self.depth_conv4(depth_feat)
        depth_feat = self.depth_conv5(depth_feat)
        depth_feat = self.global_pool(depth_feat).view(depth_feat.size(0), -1)
        
        # 特征融合
        if self.fusion_method == 'concat':
            fused = torch.cat([rgb_feat, depth_feat], dim=1)
        
        elif self.fusion_method == 'add':
            depth_feat_projected = self.depth_projection(depth_feat)
            fused = rgb_feat + depth_feat_projected
        
        elif self.fusion_method == 'attention':
            fused = self.cross_attention(rgb_feat, depth_feat)
        
        # 回归预测
        output = self.regressor(fused)
        
        # 后处理：确保非负且在合理范围
        output = torch.clamp(output, min=0, max=2000)
        
        return output


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试
if __name__ == '__main__':
    # 测试三种融合方法
    for method in ['concat', 'add', 'attention']:
        print(f"\n测试融合方法: {method}")
        print("="*50)
        
        model = DepthFusionCNN(fusion_method=method, dropout_rate=0.5)
        num_params = count_parameters(model)
        print(f"参数量: {num_params:,}")
        
        # 模拟输入
        batch_size = 4
        rgb = torch.randn(batch_size, 3, 224, 224)
        depth = torch.randn(batch_size, 1, 224, 224)
        
        # 前向传播
        output = model(rgb, depth)
        print(f"输入 - RGB: {rgb.shape}, Depth: {depth.shape}")
        print(f"输出: {output.shape}")
        print(f"输出值: {output.squeeze()}")