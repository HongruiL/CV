#!/usr/bin/env python3
"""
测试深度归一化效果
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from src.rgbd_calorie_estimator import SynchronizedTransform

def test_depth_normalization():
    """测试深度归一化是否正确"""

    # 测试单个样本
    dish_id = 'dish_1557853314'
    data_root = Path('Nutrition5K/Nutrition5K')

    rgb_path = data_root / 'train' / 'color' / dish_id / 'rgb.png'
    depth_path = data_root / 'train' / 'depth_raw' / dish_id / 'depth_raw.png'

    if not (rgb_path.exists() and depth_path.exists()):
        print(f"❌ 样本 {dish_id} 不存在，跳过测试")
        return

    rgb_img = Image.open(rgb_path).convert('RGB')
    depth_img = Image.open(depth_path)

    print(f"测试样本: {dish_id}")
    print(f"原始深度图:")
    depth_raw = np.array(depth_img, dtype=np.uint16)
    print(f"  形状: {depth_raw.shape}")
    print(f"  范围: [{depth_raw.min()}, {depth_raw.max()}]")
    print(f"  均值: {depth_raw[depth_raw > 0].mean():.2f}")

    # 应用变换
    transform = SynchronizedTransform(
        img_size=256,
        is_training=False,
        depth_max_value=4286
    )
    rgb_tensor, depth_tensor = transform(rgb_img, depth_img)

    print(f"\n归一化后深度tensor:")
    print(f"  形状: {depth_tensor.shape}")
    print(f"  范围: [{depth_tensor.min():.4f}, {depth_tensor.max():.4f}]")
    print(f"  均值: {depth_tensor.mean():.4f}")

    # 验证归一化正确性
    expected_max = depth_raw.max() / 4286
    actual_max = depth_tensor.max().item()

    print(f"\n归一化验证:")
    print(f"  预期最大值: {expected_max:.4f}")
    print(f"  实际最大值: {actual_max:.4f}")
    print(f"  误差: {abs(expected_max - actual_max):.6f}")

    if abs(expected_max - actual_max) < 0.001:
        print("✅ 归一化到 [0, 1] 成功")
    else:
        print("❌ 归一化有误")

    return True

if __name__ == '__main__':
    try:
        test_depth_normalization()
        print("\n🎉 深度归一化测试完成！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
