import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from src.dataset import Nutrition5kDataset, get_transforms
from src.resnet18_fusion_model import ResNet18Fusion, count_parameters
from src.trainer import DepthFusionTrainer


def set_seed(seed):
    """设置所有随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_single_model(seed, train_csv, val_csv, data_root,
                       batch_size, num_epochs, image_size, device):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"训练模型 - 随机种子 {seed}")
    print(f"使用 ResNet18 + Log Transform")
    print('='*60)

    # 设置随机种子
    set_seed(seed)

    # 创建数据集（启用log变换）
    train_dataset = Nutrition5kDataset(
        csv_file=train_csv,
        data_root=data_root,
        split='train',
        transform=get_transforms('train', image_size),
        use_depth=True,
        use_log_transform=True  # 启用log变换
    )

    val_dataset = Nutrition5kDataset(
        csv_file=val_csv,
        data_root=data_root,
        split='train',
        transform=get_transforms('val', image_size),
        use_depth=True,
        use_log_transform=True  # 启用log变换
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,      # ✓ 加速数据加载
        pin_memory=True,    # ✓ 加速GPU传输
        persistent_workers=True  # ✓ 保持worker进程
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # 创建ResNet18模型
    model = ResNet18Fusion(dropout_rate=0.5)
    num_params = count_parameters(model)
    print(f"模型参数量: {num_params:,}")

    # 训练
    save_dir = f'checkpoints/resnet18_log_seed{seed}'
    trainer = DepthFusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        use_log_transform=True  # 告诉trainer使用了log变换
    )

    trainer.train(num_epochs=num_epochs)

    return save_dir


def main():
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    TRAIN_CSV = DATA_ROOT / 'nutrition5k_train_clean.csv'
    BATCH_SIZE = 32  # 增大到32，更稳定的梯度估计
    NUM_EPOCHS = 80   # 减少到80，因为lr更小了
    IMAGE_SIZE = 224
    SEEDS = [42, 123, 456, 789, 2024]  # 5个模型集成效果更好

    # 打印训练配置
    print("🚀 训练配置 (保守版):")
    print(f"  模型: ResNet18 + Log Transform")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  模型数量: {len(SEEDS)}")
    print(f"  初始学习率: 0.00002 (进一步降低确保稳定)")
    print(f"  优化器: AdamW (更好的正则化)")
    print(f"  调度器: ReduceLROnPlateau (patience=5, min_lr=1e-6)")
    print(f"  损失函数: SmoothL1Loss (稳定版)")
    print(f"  深度预处理: 简单归一化 (回退到稳定版本)")
    print(f"  数据增强: 温和版本 (减少过拟合)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"将训练 {len(SEEDS)} 个ResNet18模型用于集成")

    # ✓ 统一划分数据（所有模型使用相同的训练/验证集）
    print("\n划分训练集和验证集...")
    full_df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,  # 固定种子
        shuffle=True
    )

    train_csv_path = DATA_ROOT / 'train_split.csv'
    val_csv_path = DATA_ROOT / 'val_split.csv'
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    print(f"训练集: {len(train_df)} 样本")
    print(f"验证集: {len(val_df)} 样本")

    # 训练多个模型（相同数据，不同初始化）
    model_dirs = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n开始训练第 {i}/{len(SEEDS)} 个模型")
        try:
            save_dir = train_single_model(
                seed=seed,
                train_csv=train_csv_path,
                val_csv=val_csv_path,
                data_root=DATA_ROOT,
                batch_size=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                image_size=IMAGE_SIZE,
                device=device
            )
            model_dirs.append(save_dir)
            print(f"✓ 模型 {i} 训练完成")
        except KeyboardInterrupt:
            print("\n用户中断训练")
            break
        except Exception as e:
            print(f"\n✗ 训练模型 {i} (seed={seed}) 失败: {e}")
            import traceback
            traceback.print_exc()
            # 不要继续，直接退出
            break

    # 保存模型路径列表
    if model_dirs:
        print(f"\n{'='*60}")
        print(f"成功训练 {len(model_dirs)}/{len(SEEDS)} 个模型")
        print(f"{'='*60}")
        print("模型保存路径:")
        for dir in model_dirs:
            print(f"  - {dir}/best_model.pth")

        # 保存到文件
        with open('model_paths.txt', 'w') as f:
            for dir in model_dirs:
                f.write(f"{dir}/best_model.pth\n")

        print("\n模型路径已保存到 model_paths.txt")
        print("下一步：运行 predict_ensemble_log.py 进行集成预测")
    else:
        print("\n没有成功训练任何模型！")


def test_single_model():
    """快速测试单个模型（验证稳定性改进）"""
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    TRAIN_CSV = DATA_ROOT / 'nutrition5k_train_clean.csv'
    BATCH_SIZE = 32
    NUM_EPOCHS = 30  # 增加到30个epoch，更好观察稳定性
    IMAGE_SIZE = 224

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"快速测试单个模型（稳定性验证）- 设备: {device}")
    print(f"测试配置: {NUM_EPOCHS} epochs, CosineAnnealingWarmRestarts (T_0=15)")

    # 统一划分数据
    full_df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_csv_path = DATA_ROOT / 'train_split.csv'
    val_csv_path = DATA_ROOT / 'val_split.csv'
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    # 训练单个模型
    save_dir = train_single_model(
        seed=42,
        train_csv=train_csv_path,
        val_csv=val_csv_path,
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        image_size=IMAGE_SIZE,
        device=device
    )

    print(f"\n✅ 单模型测试完成！模型保存到: {save_dir}")
    print("请检查训练日志，确认验证损失是否稳定下降")
    return save_dir


if __name__ == '__main__':
    import sys

    print("🚀 ResNet18 + Log Transform 训练脚本")
    print("=" * 50)
    print("用法:")
    print("  python train_ensemble_log.py        # 训练所有5个模型")
    print("  python train_ensemble_log.py test   # 先测试单个模型")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 快速测试单个模型
        test_single_model()
    else:
        # 完整训练所有模型
        main()
