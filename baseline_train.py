import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.dataset import create_dataloaders
from src.baseline_model import BaselineCNN

class Trainer:
    """
    模型训练器
    """
    def __init__(self, model, train_loader, val_loader, 
                 device, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 损失函数：MSE（与Kaggle评估指标一致）
        self.criterion = nn.MSELoss()
        
        # 优化器：Adam
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=0.001,
            weight_decay=1e-5  # L2正则化
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            
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
        self.patience = 10
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch in pbar:
                # 获取数据
                images = batch['image'].to(self.device)
                targets = batch['calories'].to(self.device).unsqueeze(1)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 记录
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                targets = batch['calories'].to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs):
        """完整训练流程"""
        print(f"开始训练 - {num_epochs} epochs")
        print(f"设备: {self.device}")
        print(f"训练集: {len(self.train_loader.dataset)} 样本")
        print(f"验证集: {len(self.val_loader.dataset)} 样本")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            # 计算RMSE
            train_rmse = train_loss ** 0.5
            val_rmse = val_loss ** 0.5
            
            elapsed = time.time() - start_time
            
            print(f"\nEpoch {epoch+1}/{num_epochs} ({elapsed:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} (RMSE: {train_rmse:.2f})")
            print(f"  Val Loss:   {val_loss:.4f} (RMSE: {val_rmse:.2f})")
            print(f"  LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_loss)
                print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"  验证集未改善 ({self.patience_counter}/{self.patience})")
            
            # 早停
            if self.patience_counter >= self.patience:
                print(f"\n早停触发！最佳验证loss: {self.best_val_loss:.4f}")
                break
            
            print("-" * 50)
        
        # 保存最终模型
        self.save_checkpoint('final_model.pth', epoch, val_loss)
        
        # 保存训练历史
        self.save_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        print("\n训练完成！")
        print(f"最佳验证loss: {self.best_val_loss:.4f} (RMSE: {self.best_val_loss**0.5:.2f})")
    
    def save_checkpoint(self, filename, epoch, val_loss):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def save_history(self):
        """保存训练历史"""
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 学习率曲线
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


def main():
    """主训练函数"""
    # 配置
    DATA_ROOT = Path('Nutrition5K/Nutrition5K')
    TRAIN_CSV = DATA_ROOT / 'nutrition5k_train.csv'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    IMAGE_SIZE = 224
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("\n准备数据...")
    train_loader, val_loader = create_dataloaders(
        data_root=DATA_ROOT,
        train_csv=TRAIN_CSV,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        num_workers=0,  
        image_size=IMAGE_SIZE
    )
    
    # 创建模型
    print("\n创建模型...")
    model = BaselineCNN(dropout_rate=0.5)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir='checkpoints/baseline'
    )
    
    # 开始训练
    print("\n" + "=" * 50)
    trainer.train(num_epochs=NUM_EPOCHS)
    print("=" * 50)


if __name__ == '__main__':
    main()