import torch
import torch.nn as nn
from typing import Dict, List, Optional

class ExpertNetwork(nn.Module):
    
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class MultiExpertModel(nn.Module):
    
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 num_experts: int = 4,
                 expert_names: Optional[List[str]] = None):
        super().__init__()
        
        if expert_names is None:
            expert_names = ['SBP', 'DBP', 'HR', 'GLU']
        
        self.expert_names = expert_names
        self.experts = nn.ModuleDict()
        
        for name in expert_names:
            self.experts[name] = ExpertNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=3,
                dropout=0.1
            )
    
    def forward(self, x, target_expert: Optional[str] = None):
        if target_expert is not None:
            
            return self.experts[target_expert](x)
        else:
            
            outputs = {}
            for name, expert in self.experts.items():
                outputs[name] = expert(x)
            return outputs
