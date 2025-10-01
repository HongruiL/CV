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
    BATCH_SIZE = 8
    IMAGE_SIZE = 224
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备测试数据
    print("\n准备测试数据...")
    test_dataset = Nutrition5kDataset(
        csv_file=TEST_CSV,
        data_root=DATA_ROOT,
        split='test',
        transform=get_transforms('val', IMAGE_SIZE),
        use_depth=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    # 集成不同融合方法的模型
    fusion_methods = ['concat', 'add', 'attention']
    all_predictions = []
    valid_methods = []
    
    for method in fusion_methods:
        checkpoint_path = Path(f'checkpoints/depth_fusion_{method}/best_model.pth')
        
        if not checkpoint_path.exists():
            print(f"\n⚠️  跳过 {method}: checkpoint不存在")
            continue
        
        print(f"\n加载 {method} 模型...")
        model = DepthFusionCNN(fusion_method=method, dropout_rate=0.4)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        val_loss = checkpoint.get('val_loss', 'N/A')
        print(f"  Val Loss: {val_loss}")
        
        # TTA预测
        print(f"  预测中...")
        dish_ids, predictions = predict_with_tta(model, test_loader, device)
        predictions = post_process_predictions(predictions.flatten())
        
        all_predictions.append(predictions)
        valid_methods.append(method)
        
        print(f"  预测范围: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    if len(all_predictions) == 0:
        print("\n❌ 错误：没有找到任何可用的模型！")
        return
    
    # 集成策略1：简单平均
    ensemble_simple = np.mean(all_predictions, axis=0)
    
    # 保存简单平均结果
    submission = pd.DataFrame({
        'ID': dish_ids,
        'Value': ensemble_simple
    })
    submission.to_csv('submission_ensemble.csv', index=False)
    
    print(f"\n✅ 集成预测完成！")
    print(f"使用的模型: {', '.join(valid_methods)}")
    print(f"结果已保存到: submission_ensemble.csv")
    print(f"\n集成预测统计:")
    print(f"  样本数: {len(ensemble_simple)}")
    print(f"  均值: {ensemble_simple.mean():.2f}")
    print(f"  中位数: {np.median(ensemble_simple):.2f}")
    print(f"  范围: [{ensemble_simple.min():.2f}, {ensemble_simple.max():.2f}]")
    
    # 如果有多个模型，也保存各自的预测用于分析
    if len(all_predictions) > 1:
        for i, method in enumerate(valid_methods):
            submission_single = pd.DataFrame({
                'ID': dish_ids,
                'Value': all_predictions[i]
            })
            submission_single.to_csv(f'submission_{method}_tta.csv', index=False)
            print(f"  {method} 预测已保存")

if __name__ == '__main__':
    main()

