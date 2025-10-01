import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    """组合损失：MSE + MAE"""
    def __init__(self, mse_weight=0.7, mae_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        total = self.mse_weight * mse_loss + self.mae_weight * mae_loss
        
        return total, {
            'mse': mse_loss.item(),
            'mae': mae_loss.item()
        }

