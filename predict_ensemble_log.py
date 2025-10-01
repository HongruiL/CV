import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.dataset import Nutrition5kDataset, get_transforms
from src.resnet18_fusion_model import ResNet18Fusion
import torch.nn.functional as F


def predict_with_tta(model, image, depth, n_augment=5):
    """测试时增强（TTA）预测"""
    predictions = []

    # 原始预测
    with torch.no_grad():
        pred = model(image, depth)
        predictions.append(pred)

    # 水平翻转增强
    if n_augment > 1:
        with torch.no_grad():
            flipped_rgb = torch.flip(image, dims=[3])  # 水平翻转
            flipped_depth = torch.flip(depth, dims=[3])
            pred_flipped = model(flipped_rgb, flipped_depth)
            predictions.append(pred_flipped)

    # 轻微旋转增强（可选，较耗时）
    if n_augment > 2:
        # 这里可以添加旋转增强，但为了速度暂时跳过
        pass

    # 返回平均预测
    return torch.stack(predictions).mean(0)


def predict_ensemble_log(model_paths_file='model_paths.txt',
                        output_path='submission_resnet18_log_ensemble.csv',
                        use_tta=True, tta_augment=2):
    """使用log变换训练的多个ResNet18模型集成预测"""

    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    BATCH_SIZE = 16
    IMAGE_SIZE = 224

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取模型路径
    with open(model_paths_file, 'r') as f:
        checkpoint_paths = [line.strip() for line in f.readlines()]

    print(f"使用 {len(checkpoint_paths)} 个模型集成预测")
    print(f"设备: {device}")

    # 准备测试数据
    test_color_dir = DATA_ROOT / 'test' / 'color'
    test_ids = [d.name for d in test_color_dir.iterdir() if d.is_dir()]

    test_df = pd.DataFrame({'ID': test_ids})
    test_csv_path = DATA_ROOT / 'test_ids_temp.csv'
    test_df.to_csv(test_csv_path, index=False)

    test_dataset = Nutrition5kDataset(
        csv_file=test_csv_path,
        data_root=DATA_ROOT,
        split='test',
        transform=get_transforms('val', IMAGE_SIZE),
        use_depth=True,
        use_log_transform=False  # 测试集不需要
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # 加载所有模型
    models = []
    for i, path in enumerate(checkpoint_paths, 1):
        print(f"\n加载模型 {i}/{len(checkpoint_paths)}: {path}")
        model = ResNet34Fusion(dropout_rate=0.5).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"  验证loss: {checkpoint['val_loss']:.4f}")

    # 预测
    print(f"\n开始集成预测 (TTA增强: {use_tta})...")
    all_predictions = [[] for _ in models]
    dish_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            rgb = batch['image'].to(device)
            depth = batch['depth'].to(device)

            for i, model in enumerate(models):
                if use_tta:
                    # 使用TTA增强预测
                    outputs = predict_with_tta(model, rgb, depth, n_augment=tta_augment)
                else:
                    # 普通预测
                    outputs = model(rgb, depth)

                # 逆变换：从log空间转回原始空间
                outputs = torch.expm1(outputs)  # expm1 = exp(x) - 1
                all_predictions[i].extend(outputs.cpu().numpy().flatten().tolist())

            if len(dish_ids) == 0:
                dish_ids.extend(batch['dish_id'])

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

    print(f"\n前5个预测:")
    print(submission.head())

    return submission


def predict_single_model(checkpoint_path, output_path='submission_resnet18_single.csv'):
    """使用单个ResNet18模型预测（用于测试）"""
    from src.dataset import Nutrition5kDataset, get_transforms

    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    BATCH_SIZE = 16
    IMAGE_SIZE = 224

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备测试数据
    test_color_dir = DATA_ROOT / 'test' / 'color'
    test_ids = [d.name for d in test_color_dir.iterdir() if d.is_dir()]

    test_df = pd.DataFrame({'ID': test_ids})
    test_csv_path = DATA_ROOT / 'test_ids_temp.csv'
    test_df.to_csv(test_csv_path, index=False)

    test_dataset = Nutrition5kDataset(
        csv_file=test_csv_path,
        data_root=DATA_ROOT,
        split='test',
        transform=get_transforms('val', IMAGE_SIZE),
        use_depth=True,
        use_log_transform=False
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # 加载模型
    model = ResNet18Fusion(dropout_rate=0.5).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"加载模型: {checkpoint_path}")

    # 预测
    predictions = []
    dish_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            rgb = batch['image'].to(device)
            depth = batch['depth'].to(device)

            outputs = model(rgb, depth)
            # 逆变换：从log空间转回原始空间
            outputs = torch.expm1(outputs)
            predictions.extend(outputs.cpu().numpy().flatten().tolist())
            dish_ids.extend(batch['dish_id'])

    # 裁剪负值
    predictions = np.clip(predictions, 0, None)

    # 生成提交文件
    submission = pd.DataFrame({
        'ID': dish_ids,
        'Value': predictions
    })
    submission = submission.sort_values('ID').reset_index(drop=True)
    submission.to_csv(output_path, index=False)

    print(f"\n单模型预测完成！")
    print(f"输出文件: {output_path}")
    print(f"预测统计:")
    print(f"  最小值: {np.min(predictions):.2f}")
    print(f"  最大值: {np.max(predictions):.2f}")
    print(f"  平均值: {np.mean(predictions):.2f}")
    print(f"  中位数: {np.median(predictions):.2f}")

    return submission


if __name__ == '__main__':
    # 先预测单个模型看看（可选）
    # predict_single_model('checkpoints/resnet18_log_seed42/best_model.pth')

    # 然后集成预测（回退到稳定版本，不用TTA）
    predict_ensemble_log(
        use_tta=False      # 禁用TTA，回退到简单版本
    )
