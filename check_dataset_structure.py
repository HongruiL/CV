#!/usr/bin/env python3
"""
检查数据集结构
"""
import os
from pathlib import Path

def check_dataset_structure():
    """检查数据集目录结构"""

    base_path = Path('Nutrition5K/Nutrition5K')

    print(f"基础路径: {base_path}")
    print(f"基础路径存在: {base_path.exists()}")

    # 检查训练集结构
    train_color = base_path / 'train' / 'color'
    train_depth = base_path / 'train' / 'depth_raw'

    print(f"\n训练RGB目录: {train_color}")
    print(f"训练RGB目录存在: {train_color.exists()}")

    print(f"训练深度目录: {train_depth}")
    print(f"训练深度目录存在: {train_depth.exists()}")

    # 检查样本结构
    if train_color.exists():
        sample_dirs = [d for d in train_color.iterdir() if d.is_dir()][:3]  # 只检查前3个
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            print(f"\n样本: {sample_id}")

            # 检查RGB文件
            rgb_file = sample_dir / 'rgb.png'
            print(f"  RGB文件存在: {rgb_file.exists()}")

            # 检查深度文件
            depth_file = train_depth / sample_id / 'depth_raw.png'
            print(f"  深度文件存在: {depth_file.exists()}")

            # 检查其他可能的位置
            depth_file_alt = sample_dir / 'depth_raw.png'
            if not depth_file.exists() and depth_file_alt.exists():
                print(f"  ⚠️  深度文件在错误位置: {depth_file_alt}")

    # 检查CSV文件
    csv_file = base_path / 'nutrition5k_train_clean.csv'
    print(f"\nCSV文件: {csv_file}")
    print(f"CSV文件存在: {csv_file.exists()}")

    if csv_file.exists():
        import pandas as pd
        df = pd.read_csv(csv_file)
        print(f"CSV行数: {len(df)}")
        print(f"CSV列名: {list(df.columns)}")

        # 检查前几个样本ID
        sample_ids = df['ID'].head().tolist()
        print(f"前5个样本ID: {sample_ids}")

if __name__ == '__main__':
    check_dataset_structure()
