import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from src.rgbd_calorie_estimator import RGBDCalorieEstimator, Nutrition5KDataset, SynchronizedTransform
from src.trainer import DepthFusionTrainer


def main():
    """训练高级RGB-D卡路里估算模型"""
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    TRAIN_CSV = DATA_ROOT / 'nutrition5k_train_clean.csv'
    BATCH_SIZE = 16  # 高级模型需要较小的batch size
    NUM_EPOCHS = 80  # 减少训练轮数，因为模型更复杂
    IMAGE_SIZE = 256  # 使用更大的图像尺寸

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("🚀 训练高级RGB-D卡路里估算模型（中级融合）")

    # 打印模型配置
    print("📋 模型配置:")
    print(f"  架构: 双分支ResNet + 中级融合")
    print(f"  输入尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  基础通道数: 64")
    print("=" * 60)

    # 统一划分数据（确保包含正确的列名）
    print("\n划分训练集和验证集...")
    full_df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # 确保列名正确（高级模型期望ID和Value列）
    train_df = train_df[['ID', 'Value']] if 'Value' in train_df.columns else train_df
    val_df = val_df[['ID', 'Value']] if 'Value' in val_df.columns else val_df

    train_csv_path = DATA_ROOT / 'train_split.csv'
    val_csv_path = DATA_ROOT / 'val_split.csv'
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    print(f"训练集: {len(train_df)} 样本")
    print(f"验证集: {len(val_df)} 样本")

    # 创建数据集（使用统计得出的归一化参数）
    train_transform = SynchronizedTransform(
        img_size=IMAGE_SIZE,
        is_training=True,
        depth_max_value=4286  # 从统计结果得出
    )

    val_transform = SynchronizedTransform(
        img_size=IMAGE_SIZE,
        is_training=False,
        depth_max_value=4286  # 相同参数
    )

    train_dataset = Nutrition5KDataset(
        rgb_dir=str(DATA_ROOT / 'train' / 'color'),
        depth_dir=str(DATA_ROOT / 'train' / 'depth_raw'),
        labels_csv=str(train_csv_path),
        transform=train_transform
    )

    val_dataset = Nutrition5KDataset(
        rgb_dir=str(DATA_ROOT / 'train' / 'color'),  # 注意：验证集也来自train目录
        depth_dir=str(DATA_ROOT / 'train' / 'depth_raw'),
        labels_csv=str(val_csv_path),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # 减少worker数量，避免内存问题
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 创建高级RGB-D模型
    print("\n创建高级RGB-D模型...")
    model = RGBDCalorieEstimator(
        base_channels=64,
        dropout_rate=0.5
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 使用改进的训练器
    save_dir = 'checkpoints/advanced_rgbd'
    trainer = DepthFusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        use_log_transform=False  # 高级模型使用原始空间，不需要log变换
    )

    print("\n" + "=" * 60)
    trainer.train(num_epochs=NUM_EPOCHS)
    print("=" * 60)


def test_single_advanced_model():
    """快速测试单个高级模型"""
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    TRAIN_CSV = DATA_ROOT / 'nutrition5k_train_clean.csv'
    BATCH_SIZE = 16
    NUM_EPOCHS = 20  # 先试20个epoch看看稳定性
    IMAGE_SIZE = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"快速测试高级RGB-D模型 - 设备: {device}")
    print(f"测试配置: {NUM_EPOCHS} epochs, 图像尺寸: {IMAGE_SIZE}")

    # 统一划分数据（确保包含正确的列名）
    full_df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # 确保列名正确（高级模型期望ID和Value列）
    train_df = train_df[['ID', 'Value']] if 'Value' in train_df.columns else train_df
    val_df = val_df[['ID', 'Value']] if 'Value' in val_df.columns else val_df

    train_csv_path = DATA_ROOT / 'train_split.csv'
    val_csv_path = DATA_ROOT / 'val_split.csv'
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    # 创建数据集（较小的图像尺寸用于测试）
    train_transform = SynchronizedTransform(
        img_size=224,
        is_training=True,
        depth_max_value=4286  # 测试也用相同的归一化参数
    )
    val_transform = SynchronizedTransform(
        img_size=224,
        is_training=False,
        depth_max_value=4286
    )

    train_dataset = Nutrition5KDataset(
        rgb_dir=str(DATA_ROOT / 'train' / 'color'),
        depth_dir=str(DATA_ROOT / 'train' / 'depth_raw'),
        labels_csv=str(train_csv_path),
        transform=train_transform
    )

    val_dataset = Nutrition5KDataset(
        rgb_dir=str(DATA_ROOT / 'train' / 'color'),
        depth_dir=str(DATA_ROOT / 'train' / 'depth_raw'),
        labels_csv=str(val_csv_path),
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 创建模型
    model = RGBDCalorieEstimator(base_channels=64, dropout_rate=0.5)

    # 使用改进的训练器
    save_dir = 'checkpoints/test_advanced_rgbd'
    trainer = DepthFusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        use_log_transform=False  # 高级模型使用原始空间
    )

    trainer.train(num_epochs=NUM_EPOCHS)

    print(f"\n✅ 高级模型测试完成！模型保存到: {save_dir}")
    return save_dir


if __name__ == '__main__':
    import sys

    print("🚀 高级RGB-D卡路里估算模型训练脚本")
    print("=" * 50)
    print("用法:")
    print("  python train_advanced_rgbd.py        # 训练完整模型")
    print("  python train_advanced_rgbd.py test   # 先测试单个模型")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 快速测试单个模型
        test_single_advanced_model()
    else:
        # 完整训练
        main()
