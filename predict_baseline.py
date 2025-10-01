import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset import Nutrition5kDataset, get_transforms
from src.baseline_model import BaselineCNN

def predict_test_set(model_path, data_root, output_csv='submission.csv'):
    """
    对测试集进行预测并生成Kaggle提交文件
    
    Args:
        model_path: 模型权重路径
        data_root: 数据根目录
        output_csv: 输出CSV文件名
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = BaselineCNN(dropout_rate=0.5)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型来自 epoch {checkpoint['epoch']}, 验证loss: {checkpoint['val_loss']:.4f}")
    
    # 创建测试集（假设测试集CSV只有ID列）
    test_csv = data_root / 'test_ids.csv'
    
    # 如果没有test_ids.csv，创建一个
    if not test_csv.exists():
        print("创建测试集ID列表...")
        test_dir = data_root / 'test' / 'color'
        test_ids = [d.name for d in test_dir.iterdir() if d.is_dir()]
        test_df = pd.DataFrame({'ID': test_ids})
        test_df.to_csv(test_csv, index=False)
        print(f"测试集样本数: {len(test_ids)}")
    
    # 创建测试数据集
    test_dataset = Nutrition5kDataset(
        csv_file=test_csv,
        data_root=data_root,
        split='test',
        transform=get_transforms('test', image_size=224),
        use_depth=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 预测
    print("开始预测...")
    predictions = []
    dish_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            images = batch['image'].to(device)
            outputs = model(images)
            
            # 收集预测值和ID
            predictions.extend(outputs.squeeze().cpu().numpy().tolist())
            dish_ids.extend(batch['dish_id'])
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'ID': dish_ids,
        'Value': predictions
    })
    
    # 保存
    submission_df.to_csv(output_csv, index=False)
    print(f"\n✓ 提交文件已保存: {output_csv}")
    print(f"  样本数: {len(submission_df)}")
    print(f"  预测范围: {submission_df['Value'].min():.2f} - {submission_df['Value'].max():.2f}")
    print(f"  预测均值: {submission_df['Value'].mean():.2f}")
    print(f"  预测中位数: {submission_df['Value'].median():.2f}")
    
    # 显示前几行
    print("\n前5行预测:")
    print(submission_df.head())
    
    return submission_df


def analyze_validation_predictions(model_path, data_root, val_csv):
    """
    分析验证集预测，找出最好和最差的样本
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = BaselineCNN(dropout_rate=0.5)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 创建验证数据集
    val_dataset = Nutrition5kDataset(
        csv_file=val_csv,
        data_root=data_root,
        split='train',
        transform=get_transforms('val', image_size=224),
        use_depth=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # 预测
    all_predictions = []
    all_targets = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Analyzing validation set'):
            images = batch['image'].to(device)
            targets = batch['calories'].to(device)
            outputs = model(images)
            
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_ids.extend(batch['dish_id'])
    
    # 计算误差
    import numpy as np
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    errors = np.abs(predictions - targets)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'dish_id': all_ids,
        'true_calories': targets,
        'pred_calories': predictions,
        'abs_error': errors,
        'rel_error': errors / (targets + 1e-6) * 100  # 相对误差百分比
    })
    
    # 排序
    results_df = results_df.sort_values('abs_error', ascending=False)
    
    print("\n" + "="*60)
    print("验证集错误分析")
    print("="*60)
    
    print(f"\n总体统计:")
    print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.2f}")
    print(f"  MAE: {np.mean(errors):.2f}")
    print(f"  中位数误差: {np.median(errors):.2f}")
    
    print(f"\n最差的10个预测:")
    print(results_df.head(10)[['dish_id', 'true_calories', 'pred_calories', 'abs_error']])
    
    print(f"\n最好的10个预测:")
    print(results_df.tail(10)[['dish_id', 'true_calories', 'pred_calories', 'abs_error']])
    
    # 按真实卡路里分组分析
    print(f"\n按卡路里范围分析:")
    results_df['calorie_range'] = pd.cut(results_df['true_calories'], 
                                          bins=[0, 100, 200, 300, 500, 5000],
                                          labels=['0-100', '100-200', '200-300', '300-500', '500+'])
    
    for range_name, group in results_df.groupby('calorie_range'):
        print(f"  {range_name}: 样本数={len(group)}, 平均误差={group['abs_error'].mean():.2f}")
    
    # 保存详细结果
    results_df.to_csv('validation_analysis.csv', index=False)
    print(f"\n✓ 详细分析已保存到 validation_analysis.csv")
    
    return results_df


if __name__ == '__main__':
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    MODEL_PATH = Path('checkpoints/baseline/best_model.pth')
    VAL_CSV = DATA_ROOT / 'val_split.csv'
    
    print("="*60)
    print("1. 生成测试集预测（Kaggle提交）")
    print("="*60)
    submission_df = predict_test_set(
        model_path=MODEL_PATH,
        data_root=DATA_ROOT,
        output_csv='baseline_submission.csv'
    )
    
    print("\n" + "="*60)
    print("2. 分析验证集预测")
    print("="*60)
    if VAL_CSV.exists():
        analysis_df = analyze_validation_predictions(
            model_path=MODEL_PATH,
            data_root=DATA_ROOT,
            val_csv=VAL_CSV
        )
    else:
        print("未找到验证集CSV，跳过分析")