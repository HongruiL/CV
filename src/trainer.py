# src/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F  # ← 添加这行
import torch.optim as optim
from pathlib import Path
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class DepthFusionTrainer:
    """深度融合模型训练器"""
    def __init__(self, model, train_loader, val_loader, device,
                 save_dir='checkpoints', use_log_transform=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.use_log_transform = use_log_transform  # 新增log变换标志

        # 回退到简单稳定的配置
        self.criterion = nn.SmoothL1Loss()  # 不用混合损失

        # 但评估用MSE（和Kaggle一致）
        self.mse_criterion = nn.MSELoss()
        
        # 学习率再降低一点
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.00002,  # 从0.00003降到0.00002
            weight_decay=1e-4
        )

        # 用更稳定的ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,  # 5个epoch不改善才降lr
            min_lr=1e-6
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        # 早停
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 10  # ResNet训练更慢，增加patience
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch in pbar:
                # 适配高级RGB-D模型的数据格式 (rgb_tensor, depth_tensor, calories)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    rgb, depth, targets = batch
                    rgb = rgb.to(self.device)
                    depth = depth.to(self.device)
                    targets = targets.to(self.device).unsqueeze(1)
                else:
                    # 保持原有字典格式的兼容性
                    rgb = batch['image'].to(self.device)
                    depth = batch['depth'].to(self.device)
                    targets = batch['calories'].to(self.device).unsqueeze(1)

                outputs = self.model(rgb, depth)
                loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_mse_original = 0  # 原始空间的MSE
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 适配高级RGB-D模型的数据格式
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    rgb, depth, targets = batch
                    rgb = rgb.to(self.device)
                    depth = depth.to(self.device)
                    targets = targets.to(self.device).unsqueeze(1)
                else:
                    # 保持原有字典格式的兼容性
                    rgb = batch['image'].to(self.device)
                    depth = batch['depth'].to(self.device)
                    targets = batch['calories'].to(self.device).unsqueeze(1)

                outputs = self.model(rgb, depth)

                # Log空间的loss（用于训练）
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # 原始空间的MSE（用于真实评估）
                if self.use_log_transform:
                    # 逆变换到原始空间（添加安全检查）
                    try:
                        pred_original = torch.expm1(outputs)
                        target_original = torch.expm1(targets)

                        # 检查是否有无穷大或NaN值
                        if torch.isinf(pred_original).any() or torch.isnan(pred_original).any():
                            print(f"警告：预测值包含无穷大或NaN: pred范围 [{pred_original.min().item():.2f}, {pred_original.max().item():.2f}]")
                            # 使用clip来防止溢出
                            pred_original = torch.clamp(pred_original, 0, 10000)  # 假设卡路里不会超过10000

                        if torch.isinf(target_original).any() or torch.isnan(target_original).any():
                            print(f"警告：目标值包含无穷大或NaN: target范围 [{target_original.min().item():.2f}, {target_original.max().item():.2f}]")
                            target_original = torch.clamp(target_original, 0, 10000)

                    except Exception as e:
                        print(f"逆变换错误: {e}")
                        # 如果逆变换失败，使用原始值（可能已经是log空间）
                        pred_original = outputs
                        target_original = targets
                else:
                    pred_original = outputs
                    target_original = targets

                mse_original = ((pred_original - target_original) ** 2).mean()
                total_mse_original += mse_original.item()

        return total_loss / num_batches, total_mse_original / num_batches
    
    def train(self, num_epochs):
        """完整训练流程"""
        print(f"开始训练 - {num_epochs} epochs")
        print(f"设备: {self.device}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss, val_mse_original = self.validate()  # 接收两个值

            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)  # ReduceLROnPlateau需要传入验证损失
            current_lr = self.optimizer.param_groups[0]['lr']

            if current_lr != old_lr:
                print(f"  学习率降低: {old_lr:.6f} -> {current_lr:.6f}")

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)

            train_rmse = train_loss ** 0.5
            val_rmse = val_mse_original ** 0.5  # 用原始空间MSE计算RMSE
            elapsed = time.time() - start_time

            print(f"\nEpoch {epoch+1}/{num_epochs} ({elapsed:.1f}s)")
            print(f"  Train SmoothL1: {train_loss:.4f}")
            print(f"  Val SmoothL1:   {val_loss:.4f}")
            print(f"  Val MSE (原始空间): {val_mse_original:.4f} (RMSE: {val_rmse:.2f})")  # 真实指标
            print(f"  LR: {current_lr:.6f}")

            # 用原始空间的MSE判断最佳模型
            if val_mse_original < self.best_val_loss:
                self.best_val_loss = val_mse_original
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_loss)
                print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"  验证集未改善 ({self.patience_counter}/{self.patience})")
            
            if self.patience_counter >= self.patience:
                print(f"\n早停触发！最佳验证loss: {self.best_val_loss:.4f}")
                break
            
            print("-" * 50)
        
        self.save_checkpoint('final_model.pth', epoch, val_loss)
        self.save_history()
        self.plot_training_curves()
        
        print("\n训练完成！")
        print(f"最佳验证MSE: {self.best_val_loss:.4f} (RMSE: {self.best_val_loss**0.5:.2f})")
    
    def save_checkpoint(self, filename, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def save_history(self):
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(epochs, self.history['lr'], marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        print(f"训练曲线已保存到 {self.save_dir / 'training_curves.png'}")
        plt.close()