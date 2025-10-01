import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from src.dataset import Nutrition5kDataset, get_transforms
from src.resnet_fusion_model import ResNetFusion, count_parameters
from src.trainer import DepthFusionTrainer  # 复用你原来的trainer

def main():
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    TRAIN_CSV = DATA_ROOT / 'nutrition5k_train_clean.csv'
    BATCH_SIZE = 16  # ResNet更大，用更小的batch size
    NUM_EPOCHS = 50
    IMAGE_SIZE = 224
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    print("\n准备数据...")
    full_df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)
    
    train_csv_path = DATA_ROOT / 'train_split.csv'
    val_csv_path = DATA_ROOT / 'val_split.csv'
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    train_dataset = Nutrition5kDataset(
        csv_file=train_csv_path,
        data_root=DATA_ROOT,
        split='train',
        transform=get_transforms('train', IMAGE_SIZE),
        use_depth=True
    )
    
    val_dataset = Nutrition5kDataset(
        csv_file=val_csv_path,
        data_root=DATA_ROOT,
        split='train',
        transform=get_transforms('val', IMAGE_SIZE),
        use_depth=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4)
    
    # 创建ResNet融合模型
    print("\n创建ResNet融合模型...")
    model = ResNetFusion(dropout_rate=0.5)
    num_params = count_parameters(model)
    print(f"模型参数量: {num_params:,}")
    
    # 训练
    trainer = DepthFusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir='checkpoints/resnet_fusion'
    )
    
    print("\n" + "=" * 50)
    trainer.train(num_epochs=NUM_EPOCHS)
    print("=" * 50)


if __name__ == '__main__':
    main()