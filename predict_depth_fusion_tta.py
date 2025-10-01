import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np

from src.depth_fusion_model import DepthFusionCNN
from src.dataset import Nutrition5kDataset, get_transforms
from src.tta_predict import predict_with_tta
from src.postprocess import post_process_predictions

def main():
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    TEST_CSV = DATA_ROOT / 'test_ids.csv'
    CHECKPOINT_DIR = Path('checkpoints/depth_fusion_concat')  # 修改为你的checkpoint目录
    BATCH_SIZE = 8  # TTA使用较小batch避免内存不足
    IMAGE_SIZE = 224
    FUSION_METHOD = 'concat'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    model = DepthFusionCNN(fusion_method=FUSION_METHOD, dropout_rate=0.4)
    checkpoint = torch.load(CHECKPOINT_DIR / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"已加载checkpoint - Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    # 创建测试数据集
    print("\n准备测试数据...")
    test_dataset = Nutrition5kDataset(
        csv_file=TEST_CSV,
        data_root=DATA_ROOT,
        split='test',
        transform=get_transforms('val', IMAGE_SIZE),  # 使用val transform
        use_depth=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    # TTA预测
    print("\n开始TTA预测（4个增强版本）...")
    dish_ids, predictions = predict_with_tta(model, test_loader, device)
    
    # 后处理
    print("\n后处理预测结果...")
    predictions = post_process_predictions(predictions.flatten())
    
    # 保存结果
    submission = pd.DataFrame({
        'ID': dish_ids,
        'Value': predictions
    })
    
    output_file = 'submission_tta.csv'
    submission.to_csv(output_file, index=False)
    
    print(f"\n预测完成！")
    print(f"结果已保存到: {output_file}")
    print(f"预测统计:")
    print(f"  样本数: {len(predictions)}")
    print(f"  均值: {predictions.mean():.2f}")
    print(f"  中位数: {np.median(predictions):.2f}")
    print(f"  范围: [{predictions.min():.2f}, {predictions.max():.2f}]")

if __name__ == '__main__':
    main()

