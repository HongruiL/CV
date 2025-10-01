import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms

class Nutrition5kDataset(Dataset):
    """
    Nutrition5K 数据集加载器
    
    Args:
        csv_file: CSV文件路径，包含ID和Value列
        data_root: 数据根目录
        split: 'train' 或 'test'
        transform: 图像变换
        use_depth: 是否使用深度图（默认False，baseline只用RGB）
    """
    def __init__(self, csv_file, data_root, split='train', 
                 transform=None, use_depth=False):
        self.df = pd.read_csv(csv_file)
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.use_depth = use_depth
        
        # 构建图像路径
        self.rgb_dir = self.data_root / split / 'color'
        if use_depth:
            self.depth_dir = self.data_root / split / 'depth_raw'
        
        print(f"加载 {split} 数据集: {len(self.df)} 个样本")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取样本信息
        row = self.df.iloc[idx]
        dish_id = row['ID']
        
        # 加载RGB图像
        rgb_path = self.rgb_dir / dish_id / 'rgb.png'
        rgb = Image.open(rgb_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            rgb = self.transform(rgb)
        
        # 准备返回值
        sample = {
            'image': rgb,
            'dish_id': dish_id
        }
        
        # 如果是训练集，添加标签
        if 'Value' in row:
            sample['calories'] = torch.tensor(row['Value'], dtype=torch.float32)
        
        # 如果使用深度图（baseline暂时不用）
        if self.use_depth:
            depth_path = self.depth_dir / dish_id / 'depth_raw.png'
            depth = Image.open(depth_path)
            depth = np.array(depth, dtype=np.float32) / 10000.0  # 转为米
            depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
            sample['depth'] = depth
        
        return sample


def get_transforms(split='train', image_size=224):
    """
    获取数据变换
    
    Args:
        split: 'train' 或 'val'/'test'
        image_size: 目标图像尺寸
    """
    if split == 'train':
        # 训练集：数据增强
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
            transforms.RandomRotation(degrees=15),    # ±15度旋转
            transforms.ColorJitter(                   # 色彩抖动
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # 验证集/测试集：只做基本变换
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloaders(data_root, train_csv, batch_size=32, 
                       val_split=0.2, num_workers=0, image_size=224):
    """
    创建训练集和验证集的DataLoader
    
    Args:
        data_root: 数据根目录
        train_csv: 训练CSV文件路径
        batch_size: 批次大小
        val_split: 验证集比例（0.2 = 20%）
        num_workers: 数据加载线程数
        image_size: 图像尺寸
    
    Returns:
        train_loader, val_loader
    """
    # 读取完整训练集
    full_df = pd.read_csv(train_csv)
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        full_df, 
        test_size=val_split, 
        random_state=42,
        shuffle=True
    )
    
    # 保存临时CSV
    train_csv_path = Path(train_csv).parent / 'train_split.csv'
    val_csv_path = Path(train_csv).parent / 'val_split.csv'
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"数据集划分: 训练={len(train_df)}, 验证={len(val_df)}")
    
    # 创建数据集
    train_dataset = Nutrition5kDataset(
        csv_file=train_csv_path,
        data_root=data_root,
        split='train',
        transform=get_transforms('train', image_size),
        use_depth=False  # baseline不用深度图
    )
    
    val_dataset = Nutrition5kDataset(
        csv_file=val_csv_path,
        data_root=data_root,
        split='train',  # 注意：验证集也来自train目录
        transform=get_transforms('val', image_size),
        use_depth=False
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU传输
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# 测试代码
if __name__ == '__main__':
    # 测试数据加载
    data_root = Path('Nutrition5K/Nutrition5K')
    train_csv = data_root / 'nutrition5k_train.csv'
    
    train_loader, val_loader = create_dataloaders(
        data_root=data_root,
        train_csv=train_csv,
        batch_size=16,
        val_split=0.2
    )
    
    # 获取一个batch查看
    batch = next(iter(train_loader))
    print(f"\nBatch信息:")
    print(f"  图像shape: {batch['image'].shape}")  # (B, 3, 224, 224)
    print(f"  卡路里shape: {batch['calories'].shape}")  # (B,)
    print(f"  卡路里值: {batch['calories'][:5]}")