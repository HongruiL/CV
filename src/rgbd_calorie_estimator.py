import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import random


# ===================== 模型架构 =====================

class ResidualBlock(nn.Module):
    """ResNet风格的残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 下采样层（如果需要）
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ModalityEncoder(nn.Module):
    """单模态编码器（ResNet风格）"""
    def __init__(self, in_channels=3, base_channels=64):
        super(ModalityEncoder, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差块组
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class RGBDCalorieEstimator(nn.Module):
    """双分支RGB-D卡路里估算模型（中级融合）"""
    def __init__(self, base_channels=64, dropout_rate=0.5):
        super(RGBDCalorieEstimator, self).__init__()

        # RGB分支（3通道输入）
        self.rgb_encoder = ModalityEncoder(in_channels=3, base_channels=base_channels)

        # 深度分支（1通道输入）
        self.depth_encoder = ModalityEncoder(in_channels=1, base_channels=base_channels)

        # 融合后的特征维度
        fused_channels = base_channels * 8 * 2  # 两个分支的输出拼接

        # 融合后的卷积层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_channels, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 回归头
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)  # 输出卡路里值
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, depth):
        # RGB分支编码
        rgb_features = self.rgb_encoder(rgb)

        # 深度分支编码
        depth_features = self.depth_encoder(depth)

        # 中级融合：特征拼接
        fused_features = torch.cat([rgb_features, depth_features], dim=1)

        # 融合后的卷积处理
        fused_features = self.fusion_conv(fused_features)

        # 全局平均池化
        pooled = self.avgpool(fused_features)
        pooled = torch.flatten(pooled, 1)

        # 回归预测
        calories = self.fc(pooled)

        return calories


# ===================== 数据集类 =====================

class SynchronizedTransform:
    """同步的RGB-D数据增强"""
    def __init__(self, img_size=256, is_training=True,
                 depth_max_value=4286):  # 使用统计得出的值
        self.img_size = img_size
        self.is_training = is_training
        self.depth_max_value = depth_max_value

        # RGB归一化参数（ImageNet统计）
        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, rgb_img, depth_img):
        # === RGB处理 ===
        rgb_img = F.resize(rgb_img, (self.img_size, self.img_size))

        # === 深度处理（关键修复）===
        # 1. 正确读取16-bit深度
        depth_array = np.array(depth_img, dtype=np.uint16)

        # 2. Resize深度图（使用NEAREST避免插值破坏深度值）
        depth_pil = Image.fromarray(depth_array)
        depth_pil = F.resize(
            depth_pil,
            (self.img_size, self.img_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )
        depth_array = np.array(depth_pil, dtype=np.float32)

        # 3. 统一归一化到[0, 1]
        # 使用全局max_value，所有图片尺度一致
        depth_array = np.clip(depth_array, 0, self.depth_max_value)
        depth_array = depth_array / self.depth_max_value

        # 4. 转为tensor [1, H, W]
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)

        # === 训练时数据增强 ===
        if self.is_training:
            # 同步几何变换
            # 随机水平翻转
            if random.random() > 0.5:
                rgb_img = F.hflip(rgb_img)
                depth_tensor = torch.flip(depth_tensor, dims=[2])  # 水平翻转

            # 随机旋转（-15到15度）
            angle = random.uniform(-15, 15)
            rgb_img = F.rotate(rgb_img, angle)

            # 旋转深度图（转回PIL再旋转）
            depth_pil_rot = F.to_pil_image(depth_tensor)
            depth_pil_rot = F.rotate(depth_pil_rot, angle)
            depth_tensor = F.to_tensor(depth_pil_rot)

            # 随机缩放裁剪（90%-110%）
            scale = random.uniform(0.9, 1.1)
            new_size = int(self.img_size * scale)
            rgb_img = F.resize(rgb_img, (new_size, new_size))
            rgb_img = F.center_crop(rgb_img, self.img_size)

            depth_tensor_scaled = F.resize(
                depth_tensor,
                (new_size, new_size),
                interpolation=transforms.InterpolationMode.NEAREST
            )
            depth_tensor = F.center_crop(depth_tensor_scaled, self.img_size)

        # === RGB转tensor和归一化 ===
        rgb_tensor = F.to_tensor(rgb_img)
        rgb_tensor = self.rgb_normalize(rgb_tensor)

        # === RGB光度增强（仅训练时）===
        if self.is_training:
            # 亮度调整
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                rgb_tensor = F.adjust_brightness(rgb_tensor, brightness_factor)
            # 对比度调整
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                rgb_tensor = F.adjust_contrast(rgb_tensor, contrast_factor)

            # 深度轻微噪声（模拟传感器误差）
            if random.random() > 0.5:
                noise = torch.randn_like(depth_tensor) * 0.02
                depth_tensor = torch.clamp(depth_tensor + noise, 0, 1)

        return rgb_tensor, depth_tensor


class Nutrition5KDataset(Dataset):
    """Nutrition5K数据集"""
    def __init__(self, rgb_dir, depth_dir, labels_csv, transform=None):
        """
        Args:
            rgb_dir: RGB图像目录
            depth_dir: 深度图像目录
            labels_csv: 包含文件名和卡路里标签的CSV文件
            transform: 数据增强变换
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform

        # 读取标签
        self.data = pd.read_csv(labels_csv)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取文件名和标签
        row = self.data.iloc[idx]
        img_name = row['ID']  # 使用正确的列名

        # 处理测试集（没有Value列的情况）
        if 'Value' in row:
            calories = row['Value']  # 使用正确的列名
        else:
            calories = 0.0  # 测试集的占位值

        # 读取RGB图像（与原始模型保持一致）
        rgb_path = os.path.join(self.rgb_dir, img_name, 'rgb.png')
        rgb_img = Image.open(rgb_path).convert('RGB')

        # 读取深度图像（修正路径构造）
        depth_path = os.path.join(self.depth_dir, img_name, 'depth_raw.png')
        depth_img = Image.open(depth_path)

        # 应用变换
        if self.transform:
            rgb_tensor, depth_tensor = self.transform(rgb_img, depth_img)
        else:
            rgb_tensor = F.to_tensor(rgb_img)
            depth_tensor = F.to_tensor(depth_img)

        calories = torch.tensor(calories, dtype=torch.float32)

        return rgb_tensor, depth_tensor, calories


# ===================== 训练和评估函数 =====================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_mae = 0.0

    pbar = tqdm(dataloader, desc='Training')
    for rgb, depth, targets in pbar:
        rgb = rgb.to(device)
        depth = depth.to(device)
        targets = targets.to(device).unsqueeze(1)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(rgb, depth)
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # 统计
        running_loss += loss.item() * rgb.size(0)
        running_mse += loss.item() * rgb.size(0)
        running_mae += torch.abs(outputs - targets).mean().item() * rgb.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.2f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mse = running_mse / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    epoch_rmse = np.sqrt(epoch_mse)

    return epoch_loss, epoch_rmse, epoch_mae


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_mae = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for rgb, depth, targets in pbar:
            rgb = rgb.to(device)
            depth = depth.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(rgb, depth)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * rgb.size(0)
            running_mse += loss.item() * rgb.size(0)
            running_mae += torch.abs(outputs - targets).mean().item() * rgb.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.2f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mse = running_mse / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    epoch_rmse = np.sqrt(epoch_mse)

    return epoch_loss, epoch_rmse, epoch_mae


# ===================== 主训练脚本 =====================

def main():
    # 超参数设置
    config = {
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'base_channels': 64,
        'dropout_rate': 0.5,
        'img_size': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'save_dir': 'checkpoints'
    }

    print(f"使用设备: {config['device']}")

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # 准备数据集和数据加载器
    train_transform = SynchronizedTransform(img_size=config['img_size'], is_training=True)
    val_transform = SynchronizedTransform(img_size=config['img_size'], is_training=False)

    train_dataset = Nutrition5KDataset(
        rgb_dir='data/train/rgb',
        depth_dir='data/train/depth',
        labels_csv='data/train_labels.csv',
        transform=train_transform
    )

    val_dataset = Nutrition5KDataset(
        rgb_dir='data/val/rgb',
        depth_dir='data/val/depth',
        labels_csv='data/val_labels.csv',
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 创建模型
    model = RGBDCalorieEstimator(
        base_channels=config['base_channels'],
        dropout_rate=config['dropout_rate']
    ).to(config['device'])

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # 训练循环
    best_val_mse = float('inf')
    patience_counter = 0
    max_patience = 15

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 60)

        # 训练
        train_loss, train_rmse, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )

        # 验证
        val_loss, val_rmse, val_mae = validate(
            model, val_loader, criterion, config['device']
        )

        # 打印结果
        print(f"训练 - Loss: {train_loss:.2f}, RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
        print(f"验证 - Loss: {val_loss:.2f}, RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}")

        # 更新学习率
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_mse:
            best_val_mse = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': val_loss,
                'val_rmse': val_rmse,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"✓ 保存最佳模型 (MSE: {val_loss:.2f}, RMSE: {val_rmse:.2f})")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= max_patience:
            print(f"\n早停触发！{max_patience}个epoch没有改进。")
            break

    print(f"\n训练完成！最佳验证MSE: {best_val_mse:.2f}, RMSE: {np.sqrt(best_val_mse):.2f}")


if __name__ == '__main__':
    main()
