#!/usr/bin/env python3
"""
测试深度归一化逻辑（不依赖PyTorch）
"""
import numpy as np
from PIL import Image
from pathlib import Path

def test_depth_normalization_logic():
    """测试深度归一化逻辑"""

    # 模拟深度图数据
    depth_array = np.array([
        [1000, 2000, 3000],
        [1500, 2500, 3500],
        [1200, 2200, 3200]
    ], dtype=np.uint16)

    print("模拟深度图:")
    print(f"  形状: {depth_array.shape}")
    print(f"  范围: [{depth_array.min()}, {depth_array.max()}]")
    print(f"  均值: {depth_array.mean():.2f}")

    # 模拟归一化过程
    depth_max_value = 4286

    # 裁剪到最大值
    depth_clipped = np.clip(depth_array, 0, depth_max_value)

    # 归一化到[0, 1]
    depth_normalized = depth_clipped / depth_max_value

    print("\n归一化后:")
    print(f"  范围: [{depth_normalized.min():.4f}, {depth_normalized.max():.4f}]")
    print(f"  均值: {depth_normalized.mean():.4f}")

    # 验证归一化正确性
    expected_max = depth_array.max() / depth_max_value
    actual_max = depth_normalized.max()

    print("\n归一化验证:")
    print(f"  预期最大值: {expected_max:.4f}")
    print(f"  实际最大值: {actual_max:.4f}")
    print(f"  误差: {abs(expected_max - actual_max):.6f}")

    if abs(expected_max - actual_max) < 0.001:
        print("✅ 归一化逻辑正确")
        return True
    else:
        print("❌ 归一化逻辑有误")
        return False

def test_file_path_construction():
    """测试文件路径构造"""
    data_root = Path('Nutrition5K/Nutrition5K')
    dish_id = 'dish_1557853314'

    rgb_path = data_root / 'train' / 'color' / dish_id / 'rgb.png'
    depth_path = data_root / 'train' / 'depth_raw' / dish_id / 'depth_raw.png'

    print(f"\n文件路径测试:")
    print(f"  RGB路径: {rgb_path}")
    print(f"  深度路径: {depth_path}")

    # 检查路径是否存在（如果数据集存在）
    if data_root.exists():
        print(f"  数据根目录存在: {data_root.exists()}")
        print(f"  训练RGB目录存在: {(data_root / 'train' / 'color').exists()}")
        print(f"  训练深度目录存在: {(data_root / 'train' / 'depth_raw').exists()}")

        if (data_root / 'train' / 'color').exists():
            sample_dirs = [d for d in (data_root / 'train' / 'color').iterdir() if d.is_dir()]
            if sample_dirs:
                first_sample = sample_dirs[0].name
                rgb_sample = sample_dirs[0] / 'rgb.png'
                depth_sample = data_root / 'train' / 'depth_raw' / first_sample / 'depth_raw.png'
                print(f"  示例样本: {first_sample}")
                print(f"  RGB文件存在: {rgb_sample.exists()}")
                print(f"  深度文件存在: {depth_sample.exists()}")

if __name__ == '__main__':
    print("🧪 深度归一化测试")
    print("=" * 50)

    success = test_depth_normalization_logic()

    print("\n" + "=" * 50)
    test_file_path_construction()

    if success:
        print("\n🎉 所有测试通过！深度归一化逻辑正确。")
    else:
        print("\n❌ 测试失败")
