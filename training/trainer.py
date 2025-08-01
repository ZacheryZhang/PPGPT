import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Any
from tqdm import tqdm

class Trainer:
    
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: str = 'cuda',
                 patience: int = 20):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self) -> Dict[str, float]:
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            batch = [item.to(self.device) for item in batch]
            
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch)
            loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'train_loss': total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = [item.to(self.device) for item in batch]
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return {'val_loss': total_loss / num_batches}
    
    def compute_loss(self, batch) -> torch.Tensor:
        
        raise NotImplementedError
    
    def train(self, epochs: int) -> Dict[str, list]:
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            
            train_metrics = self.train_epoch()
            
            
            val_metrics = self.validate()
            
            
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.patience_counter = 0
                
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        return history

class DualStreamTrainer(Trainer):
    
    
    def compute_loss(self, batch) -> torch.Tensor:
        if len(batch) == 2:
            signal, image = batch
            output, signal_feat, image_feat = self.model(signal, image)
            
            loss = self.contrastive_loss(signal_feat, image_feat)
        else:
            signal, image, label = batch
            output, _, _ = self.model(signal, image)
            loss = self.criterion(output, label)
        
        return loss
    
    def contrastive_loss(self, feat1, feat2, temperature=0.07):
        
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        similarity = torch.mm(feat1, feat2.t()) / temperature
        labels = torch.arange(feat1.size(0)).to(self.device)
        
        loss = F.cross_entropy(similarity, labels)
        return loss

class MultiExpertTrainer(Trainer):
    
    
    def compute_loss(self, batch) -> torch.Tensor:
        features, labels, masks = batch
        outputs = self.model(features)
        
        total_loss = 0.0
        valid_tasks = 0
        
        for i, expert_name in enumerate(self.model.expert_names):
            task_mask = masks[:, i]  
            if task_mask.sum() > 0:  
                task_output = outputs[expert_name][task_mask.bool()]
                task_label = labels[:, i][task_mask.bool()]
                task_loss = self.criterion(task_output.squeeze(), task_label)
                total_loss += task_loss
                valid_tasks += 1
        
        return total_loss / max(valid_tasks, 1)
