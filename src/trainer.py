# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class DepthFusionTrainer:
    """深度融合模型训练器"""
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 训练用SmoothL1Loss（更鲁棒）
        self.criterion = nn.SmoothL1Loss()

        # 但评估用MSE（和Kaggle一致）
        self.mse_criterion = nn.MSELoss()
        
        # 优化器：Adam
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=0.0001,  # ResNet用更小的学习率
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
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
        total_mse = 0  # 新增MSE评估
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                rgb = batch['image'].to(self.device)
                depth = batch['depth'].to(self.device)
                targets = batch['calories'].to(self.device).unsqueeze(1)

                outputs = self.model(rgb, depth)

                # SmoothL1Loss（用于训练参考）
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # MSE（用于真实评估）
                mse = self.mse_criterion(outputs, targets)
                total_mse += mse.item()

        return total_loss / num_batches, total_mse / num_batches  # 返回两个值
    
    def train(self, num_epochs):
        """完整训练流程"""
        print(f"开始训练 - {num_epochs} epochs")
        print(f"设备: {self.device}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss, val_mse = self.validate()  # 接收两个值

            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            if current_lr != old_lr:
                print(f"  学习率降低: {old_lr:.6f} -> {current_lr:.6f}")

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)

            train_rmse = train_loss ** 0.5
            val_rmse = val_mse ** 0.5  # 用MSE计算RMSE
            elapsed = time.time() - start_time

            print(f"\nEpoch {epoch+1}/{num_epochs} ({elapsed:.1f}s)")
            print(f"  Train SmoothL1: {train_loss:.4f}")
            print(f"  Val SmoothL1:   {val_loss:.4f}")
            print(f"  Val MSE:        {val_mse:.4f} (RMSE: {val_rmse:.2f})")  # 真实指标
            print(f"  LR: {current_lr:.6f}")

            # 用MSE判断最佳模型
            if val_mse < self.best_val_loss:
                self.best_val_loss = val_mse
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