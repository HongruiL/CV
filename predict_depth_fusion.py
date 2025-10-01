import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from src.dataset import Nutrition5kDataset, get_transforms
from src.depth_fusion_model import DepthFusionCNN

def predict_test_set(model_path, data_root, output_csv='depth_fusion_submission.csv'):
    """
    使用深度融合模型对测试集进行预测
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取融合方法
    fusion_method = checkpoint.get('fusion_method', 'concat')
    print(f"融合方法: {fusion_method}")
    
    model = DepthFusionCNN(fusion_method=fusion_method, dropout_rate=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型来自 epoch {checkpoint['epoch']}, 验证loss: {checkpoint['val_loss']:.4f}")
    print(f"验证RMSE: {checkpoint['val_loss']**0.5:.2f}")
    
    # 创建测试集ID列表
    test_csv = data_root / 'test_ids.csv'
    if not test_csv.exists():
        print("\n创建测试集ID列表...")
        test_dir = data_root / 'test' / 'color'
        test_ids = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
        test_df = pd.DataFrame({'ID': test_ids})
        test_df.to_csv(test_csv, index=False)
        print(f"测试集样本数: {len(test_ids)}")
    
    # 创建测试数据集（use_depth=True）
    test_dataset = Nutrition5kDataset(
        csv_file=test_csv,
        data_root=data_root,
        split='test',
        transform=get_transforms('test', image_size=224),
        use_depth=True  # 关键：启用深度图
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 预测
    print("\n开始预测...")
    predictions = []
    dish_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            rgb = batch['image'].to(device)
            depth = batch['depth'].to(device)
            
            # 双输入前向传播
            outputs = model(rgb, depth)
            
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
    
    # 对比baseline统计
    print(f"\n与训练集统计对比（训练集均值237，中位数187）:")
    print(f"  测试集预测均值偏差: {submission_df['Value'].mean() - 237:.2f}")
    
    print("\n前5行预测:")
    print(submission_df.head())
    
    return submission_df


def analyze_validation_predictions(model_path, data_root, val_csv):
    """
    分析验证集预测，对比baseline
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    fusion_method = checkpoint.get('fusion_method', 'concat')
    
    model = DepthFusionCNN(fusion_method=fusion_method, dropout_rate=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 创建验证数据集
    val_dataset = Nutrition5kDataset(
        csv_file=val_csv,
        data_root=data_root,
        split='train',
        transform=get_transforms('val', image_size=224),
        use_depth=True
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
        for batch in tqdm(val_loader, desc='Analyzing validation'):
            rgb = batch['image'].to(device)
            depth = batch['depth'].to(device)
            targets = batch['calories'].to(device)
            
            outputs = model(rgb, depth)
            
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_ids.extend(batch['dish_id'])
    
    # 计算误差
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    errors = np.abs(predictions - targets)
    
    results_df = pd.DataFrame({
        'dish_id': all_ids,
        'true_calories': targets,
        'pred_calories': predictions,
        'abs_error': errors,
        'rel_error': errors / (targets + 1e-6) * 100
    })
    
    results_df = results_df.sort_values('abs_error', ascending=False)
    
    print("\n" + "="*60)
    print("深度融合模型 - 验证集错误分析")
    print("="*60)
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(errors)
    
    print(f"\n总体统计:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  中位数误差: {np.median(errors):.2f}")
    
    # 读取baseline结果对比
    baseline_analysis = Path('validation_analysis.csv')
    if baseline_analysis.exists():
        baseline_df = pd.read_csv(baseline_analysis)
        baseline_rmse = np.sqrt(np.mean(baseline_df['abs_error']**2))
        
        print(f"\n与Baseline对比:")
        print(f"  Baseline RMSE: {baseline_rmse:.2f}")
        print(f"  深度融合 RMSE: {rmse:.2f}")
        print(f"  改进: {baseline_rmse - rmse:.2f} ({(baseline_rmse - rmse)/baseline_rmse*100:.1f}%)")
        
        if rmse > baseline_rmse:
            print(f"  ⚠️ 警告: 深度融合模型表现不如baseline")
    
    print(f"\n最差的10个预测:")
    print(results_df.head(10)[['dish_id', 'true_calories', 'pred_calories', 'abs_error']])
    
    print(f"\n最好的10个预测:")
    print(results_df.tail(10)[['dish_id', 'true_calories', 'pred_calories', 'abs_error']])
    
    # 按卡路里范围分析
    results_df['calorie_range'] = pd.cut(
        results_df['true_calories'], 
        bins=[0, 100, 200, 300, 500, 5000],
        labels=['0-100', '100-200', '200-300', '300-500', '500+']
    )
    
    print(f"\n按卡路里范围分析:")
    for range_name, group in results_df.groupby('calorie_range'):
        avg_error = group['abs_error'].mean()
        print(f"  {range_name}: 样本数={len(group)}, 平均误差={avg_error:.2f}")
    
    # 保存结果
    results_df.to_csv('depth_fusion_validation_analysis.csv', index=False)
    print(f"\n✓ 详细分析已保存到 depth_fusion_validation_analysis.csv")
    
    return results_df


if __name__ == '__main__':
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    MODEL_PATH = Path('checkpoints/depth_fusion_concat/best_model.pth')
    VAL_CSV = DATA_ROOT / 'val_split.csv'
    
    print("="*60)
    print("深度融合模型预测")
    print("="*60)
    
    # 1. 生成测试集预测
    print("\n1. 生成测试集预测（Kaggle提交）")
    print("-"*60)
    submission_df = predict_test_set(
        model_path=MODEL_PATH,
        data_root=DATA_ROOT,
        output_csv='depth_fusion_submission.csv'
    )
    
    # 2. 分析验证集
    print("\n" + "="*60)
    print("2. 分析验证集预测")
    print("-"*60)
    if VAL_CSV.exists():
        analysis_df = analyze_validation_predictions(
            model_path=MODEL_PATH,
            data_root=DATA_ROOT,
            val_csv=VAL_CSV
        )
    else:
        print("未找到验证集CSV")
    
    print("\n" + "="*60)
    print("完成！请将 depth_fusion_submission.csv 提交到Kaggle")
    print("="*60)