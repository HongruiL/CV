import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from src.rgbd_calorie_estimator import RGBDCalorieEstimator, Nutrition5KDataset, SynchronizedTransform


def predict_with_tta(model, rgb, depth, n_augment=5):
    """测试时增强（TTA）预测"""
    predictions = []

    # 原始预测
    with torch.no_grad():
        pred = model(rgb, depth)
        predictions.append(pred)

    # 水平翻转增强
    if n_augment > 1:
        with torch.no_grad():
            flipped_rgb = torch.flip(rgb, dims=[3])  # 水平翻转
            flipped_depth = torch.flip(depth, dims=[3])
            pred_flipped = model(flipped_rgb, flipped_depth)
            predictions.append(pred_flipped)

    # 垂直翻转增强
    if n_augment > 2:
        with torch.no_grad():
            flipped_rgb_v = torch.flip(rgb, dims=[2])  # 垂直翻转
            flipped_depth_v = torch.flip(depth, dims=[2])
            pred_flipped_v = model(flipped_rgb_v, flipped_depth_v)
            predictions.append(pred_flipped_v)

    # 同时水平垂直翻转
    if n_augment > 3:
        with torch.no_grad():
            flipped_rgb_hv = torch.flip(rgb, dims=[2, 3])  # 水平垂直翻转
            flipped_depth_hv = torch.flip(depth, dims=[2, 3])
            pred_flipped_hv = model(flipped_rgb_hv, flipped_depth_hv)
            predictions.append(pred_flipped_hv)

    # 返回平均预测
    return torch.stack(predictions).mean(0)


def predict_ensemble_advanced_rgbd(model_paths, output_path='submission_advanced_rgbd_ensemble.csv',
                                  use_tta=True, tta_augment=2, batch_size=16, img_size=256):
    """使用多个Advanced RGBD模型集成预测"""

    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"使用 {len(model_paths)} 个模型集成预测")
    print(f"设备: {device}")
    print(f"使用TTA增强: {use_tta}, 增强次数: {tta_augment}")

    # 准备测试数据
    test_csv_path = DATA_ROOT / 'test_ids.csv'
    test_transform = SynchronizedTransform(img_size=img_size, is_training=False)

    # 读取测试标签以获取ID列表
    test_labels_df = pd.read_csv(test_csv_path)

    test_dataset = Nutrition5KDataset(
        rgb_dir=str(DATA_ROOT / 'test' / 'color'),
        depth_dir=str(DATA_ROOT / 'test' / 'depth_raw'),
        labels_csv=str(test_csv_path),
        transform=test_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # 加载所有模型
    models = []
    for i, path in enumerate(model_paths, 1):
        print(f"\n加载模型 {i}/{len(model_paths)}: {path}")

        # 根据模型路径推断参数
        if 'advanced_rgbd' in str(path):
            base_channels = 64
            dropout_rate = 0.5
        else:
            # 默认参数
            base_channels = 64
            dropout_rate = 0.5

        model = RGBDCalorieEstimator(base_channels=base_channels, dropout_rate=dropout_rate).to(device)
        checkpoint = torch.load(path, map_location=device)

        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        models.append(model)

        if 'val_loss' in checkpoint:
            print(f"  验证loss: {checkpoint['val_loss']:.4f}")

    # 预测
    print(f"\n开始集成预测...")
    all_predictions = [[] for _ in models]
    dish_ids = []

    with torch.no_grad():
        for batch_idx, (rgb, depth, _) in enumerate(tqdm(test_loader, desc='Predicting')):
            rgb = rgb.to(device)
            depth = depth.to(device)

            for i, model in enumerate(models):
                if use_tta:
                    # 使用TTA增强预测
                    outputs = predict_with_tta(model, rgb, depth, n_augment=tta_augment)
                else:
                    # 普通预测
                    outputs = model(rgb, depth)

                all_predictions[i].extend(outputs.cpu().numpy().flatten().tolist())

            # 获取对应的ID
            batch_size_actual = rgb.size(0)
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + batch_size_actual

            if end_idx <= len(test_labels_df):
                batch_ids = test_labels_df.iloc[start_idx:end_idx]['ID'].tolist()
                dish_ids.extend(batch_ids)

    # 平均集成
    ensemble_predictions = np.mean(all_predictions, axis=0)

    # 裁剪负值
    ensemble_predictions = np.clip(ensemble_predictions, 0, None)

    # 生成提交文件
    submission = pd.DataFrame({
        'ID': dish_ids,
        'Value': ensemble_predictions
    })
    submission = submission.sort_values('ID').reset_index(drop=True)
    submission.to_csv(output_path, index=False)

    print(f"\n集成预测完成！")
    print(f"输出文件: {output_path}")
    print(f"预测统计:")
    print(f"  最小值: {ensemble_predictions.min():.2f}")
    print(f"  最大值: {ensemble_predictions.max():.2f}")
    print(f"  平均值: {ensemble_predictions.mean():.2f}")
    print(f"  中位数: {np.median(ensemble_predictions):.2f}")

    print("\n前5个预测:")
    print(submission.head())

    return submission


def predict_single_advanced_rgbd(model_path='checkpoints/advanced_rgbd/best_model.pth',
                                output_path='submission_advanced_rgbd.csv',
                                use_tta=False, tta_augment=2, batch_size=16, img_size=256):
    """使用单个Advanced RGBD模型进行预测"""

    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"使用高级RGB-D模型预测 - 设备: {device}")
    print(f"使用TTA增强: {use_tta}, 增强次数: {tta_augment}")

    # 准备测试数据
    test_csv_path = DATA_ROOT / 'test_ids.csv'
    test_transform = SynchronizedTransform(img_size=img_size, is_training=False)

    # 读取测试标签以获取ID列表
    test_labels_df = pd.read_csv(test_csv_path)

    test_dataset = Nutrition5KDataset(
        rgb_dir=str(DATA_ROOT / 'test' / 'color'),
        depth_dir=str(DATA_ROOT / 'test' / 'depth_raw'),
        labels_csv=str(test_csv_path),
        transform=test_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # 加载模型
    print(f"\n加载模型: {model_path}")

    # 根据模型路径推断参数
    if 'advanced_rgbd' in str(model_path):
        base_channels = 64
        dropout_rate = 0.5
    else:
        # 默认参数
        base_channels = 64
        dropout_rate = 0.5

    model = RGBDCalorieEstimator(base_channels=base_channels, dropout_rate=dropout_rate).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # 处理不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    if 'val_loss' in checkpoint:
        print(f"模型来自 epoch {checkpoint.get('epoch', '未知')}, 验证MSE: {checkpoint['val_loss']:.2f}")

    # 预测
    print("\n开始预测...")
    predictions = []
    dish_ids = []

    with torch.no_grad():
        for batch_idx, (rgb, depth, _) in enumerate(tqdm(test_loader, desc='Predicting')):
            rgb = rgb.to(device)
            depth = depth.to(device)

            if use_tta:
                # 使用TTA增强预测
                outputs = predict_with_tta(model, rgb, depth, n_augment=tta_augment)
            else:
                # 普通预测
                outputs = model(rgb, depth)

            predictions.extend(outputs.cpu().numpy().flatten().tolist())

            # 获取对应的ID
            batch_size_actual = rgb.size(0)
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + batch_size_actual

            if end_idx <= len(test_labels_df):
                batch_ids = test_labels_df.iloc[start_idx:end_idx]['ID'].tolist()
                dish_ids.extend(batch_ids)

    # 裁剪负值
    predictions = np.clip(predictions, 0, None)

    # 生成提交文件
    submission = pd.DataFrame({
        'ID': dish_ids,
        'Value': predictions
    })
    submission = submission.sort_values('ID').reset_index(drop=True)
    submission.to_csv(output_path, index=False)

    print(f"\n预测完成！")
    print(f"输出文件: {output_path}")
    print(f"预测统计:")
    print(f"  最小值: {np.min(predictions):.2f}")
    print(f"  最大值: {np.max(predictions):.2f}")
    print(f"  平均值: {np.mean(predictions):.2f}")
    print(f"  中位数: {np.median(predictions):.2f}")

    print("\n前5个预测:")
    print(submission.head())

    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced RGBD模型预测')
    parser.add_argument('--model_paths', nargs='+', default=['checkpoints/advanced_rgbd/best_model.pth'],
                        help='模型路径列表（用于集成）')
    parser.add_argument('--output', default='submission_advanced_rgbd_ensemble.csv',
                        help='输出文件路径')
    parser.add_argument('--single_model', action='store_true',
                        help='使用单个模型预测（而不是集成）')
    parser.add_argument('--no_tta', action='store_true',
                        help='禁用TTA增强')
    parser.add_argument('--tta_augment', type=int, default=2,
                        help='TTA增强次数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批处理大小')
    parser.add_argument('--img_size', type=int, default=256,
                        help='图像尺寸')

    args = parser.parse_args()

    use_tta = not args.no_tta

    if args.single_model:
        # 单模型预测
        predict_single_advanced_rgbd(
            model_path=args.model_paths[0],
            output_path=args.output,
            use_tta=use_tta,
            tta_augment=args.tta_augment,
            batch_size=args.batch_size,
            img_size=args.img_size
        )
    else:
        # 集成预测
        predict_ensemble_advanced_rgbd(
            model_paths=args.model_paths,
            output_path=args.output,
            use_tta=use_tta,
            tta_augment=args.tta_augment,
            batch_size=args.batch_size,
            img_size=args.img_size
        )
