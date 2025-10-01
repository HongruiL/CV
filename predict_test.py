import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import Nutrition5kDataset, get_transforms
from src.resnet_fusion_model import ResNetFusion

def predict_test(checkpoint_path='checkpoints/resnet_fusion/best_model.pth',
                 output_path='submission.csv'):
    """
    在测试集上预测并生成Kaggle提交文件
    
    Args:
        checkpoint_path: 训练好的模型路径
        output_path: 输出的提交文件路径
    """
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    BATCH_SIZE = 16
    IMAGE_SIZE = 224
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 创建测试集ID列表
    print("\n准备测试集...")
    test_color_dir = DATA_ROOT / 'test' / 'color'
    
    if not test_color_dir.exists():
        raise FileNotFoundError(f"测试集目录不存在: {test_color_dir}")
    
    # 获取所有测试样本的ID
    test_ids = [d.name for d in test_color_dir.iterdir() if d.is_dir()]
    print(f"找到 {len(test_ids)} 个测试样本")
    
    # 创建临时CSV（只有ID列，没有Value列）
    test_df = pd.DataFrame({'ID': test_ids})
    test_csv_path = DATA_ROOT / 'test_ids_temp.csv'
    test_df.to_csv(test_csv_path, index=False)
    
    # 2. 创建测试数据集
    test_dataset = Nutrition5kDataset(
        csv_file=test_csv_path,
        data_root=DATA_ROOT,
        split='test',
        transform=get_transforms('val', IMAGE_SIZE),  # 测试用val transform
        use_depth=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  # 保持顺序
        num_workers=0
    )
    
    # 3. 加载模型
    print("\n加载模型...")
    model = ResNetFusion(dropout_rate=0.5).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"加载模型: {checkpoint_path}")
    print(f"训练时最佳验证loss: {checkpoint['val_loss']:.4f}")
    
    # 4. 预测
    print("\n开始预测...")
    predictions = []
    dish_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            rgb = batch['image'].to(device)
            depth = batch['depth'].to(device)
            
            outputs = model(rgb, depth)
            
            predictions.extend(outputs.cpu().numpy().flatten().tolist())
            dish_ids.extend(batch['dish_id'])
    
    # 5. 生成提交文件
    print("\n生成提交文件...")
    submission = pd.DataFrame({
        'ID': dish_ids,
        'Value': predictions
    })
    
    # 确保顺序和格式正确
    submission = submission.sort_values('ID').reset_index(drop=True)
    submission.to_csv(output_path, index=False)
    
    # 6. 统计信息
    print(f"\n预测完成！")
    print(f"输出文件: {output_path}")
    print(f"样本数量: {len(predictions)}")
    print(f"预测值统计:")
    print(f"  最小值: {min(predictions):.2f}")
    print(f"  最大值: {max(predictions):.2f}")
    print(f"  平均值: {sum(predictions)/len(predictions):.2f}")
    print(f"  中位数: {sorted(predictions)[len(predictions)//2]:.2f}")
    
    # 检查异常值
    negative_count = sum(1 for p in predictions if p < 0)
    if negative_count > 0:
        print(f"\n⚠️  警告: 有 {negative_count} 个负值预测")
        print("   建议将负值裁剪为0")
    
    # 显示前5个预测示例
    print(f"\n前5个预测示例:")
    print(submission.head())
    
    return submission


def predict_with_ensemble(checkpoint_paths, output_path='submission_ensemble.csv'):
    """
    使用多个模型集成预测
    
    Args:
        checkpoint_paths: 模型路径列表，例如
            ['checkpoints/resnet_fusion_seed42/best_model.pth',
             'checkpoints/resnet_fusion_seed123/best_model.pth',
             'checkpoints/resnet_fusion_seed456/best_model.pth']
        output_path: 输出文件路径
    """
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    BATCH_SIZE = 16
    IMAGE_SIZE = 224
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用 {len(checkpoint_paths)} 个模型集成")
    
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
        use_depth=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    # 加载所有模型
    models = []
    for i, path in enumerate(checkpoint_paths, 1):
        print(f"\n加载模型 {i}/{len(checkpoint_paths)}: {path}")
        model = ResNetFusion(dropout_rate=0.5).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    
    # 预测
    print("\n开始集成预测...")
    all_predictions = [[] for _ in models]
    dish_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            rgb = batch['image'].to(device)
            depth = batch['depth'].to(device)
            
            # 每个模型的预测
            for i, model in enumerate(models):
                outputs = model(rgb, depth)
                all_predictions[i].extend(outputs.cpu().numpy().flatten().tolist())
            
            if len(dish_ids) == 0:  # 只记录一次
                dish_ids.extend(batch['dish_id'])
    
    # 平均多个模型的预测
    import numpy as np
    ensemble_predictions = np.mean(all_predictions, axis=0).tolist()
    
    # 生成提交文件
    submission = pd.DataFrame({
        'ID': dish_ids,
        'Value': ensemble_predictions
    })
    submission = submission.sort_values('ID').reset_index(drop=True)
    submission.to_csv(output_path, index=False)
    
    print(f"\n集成预测完成！")
    print(f"输出文件: {output_path}")
    print(f"预测值范围: [{min(ensemble_predictions):.2f}, {max(ensemble_predictions):.2f}]")
    
    return submission


if __name__ == '__main__':
    # 单模型预测
    predict_test(
        checkpoint_path='checkpoints/resnet_fusion/best_model.pth',
        output_path='submission_resnet_fusion.csv'
    )
    
    # 如果有多个模型，使用集成预测
    # predict_with_ensemble(
    #     checkpoint_paths=[
    #         'checkpoints/resnet_fusion_seed42/best_model.pth',
    #         'checkpoints/resnet_fusion_seed123/best_model.pth',
    #         'checkpoints/resnet_fusion_seed456/best_model.pth'
    #     ],
    #     output_path='submission_ensemble.csv'
    # )