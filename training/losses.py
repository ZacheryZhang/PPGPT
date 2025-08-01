import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class AutomaticWeightedLoss(nn.Module):
    
    
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        
        weighted_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss += precision * loss + self.log_vars[i]
        
        return weighted_loss / self.num_tasks

class ContrastiveLoss(nn.Module):
    
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        
        
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        batch_size = features1.size(0)
        
        
        similarity_matrix = torch.mm(features1, features2.t()) / self.temperature
        
        
        labels = torch.arange(batch_size).to(features1.device)
        
        
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class FocalLoss(nn.Module):
    
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.mse_loss(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

class HuberLoss(nn.Module):
    
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        residual = torch.abs(inputs - targets)
        condition = residual < self.delta
        small_res = 0.5 * residual ** 2
        large_res = self.delta * residual - 0.5 * self.delta ** 2
        return torch.where(condition, small_res, large_res).mean()
