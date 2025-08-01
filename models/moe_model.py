import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math

from .base_model import BaseModel

class Expert(nn.Module):
    
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class NoisyTopKGating(nn.Module):
    
    
    def __init__(self, 
                 input_dim: int, 
                 num_experts: int, 
                 top_k: int = 2,
                 noise_epsilon: float = 1e-2):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        
        
        self.w_gate = nn.Linear(input_dim, num_experts)
        self.w_noise = nn.Linear(input_dim, num_experts)
        
        self.softplus = nn.Softplus()
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
    
    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)
    
    def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
        
        return (gates > 0).sum(0)
    
    def _prob_in_top_k(self, 
                      clean_values: torch.Tensor, 
                      noisy_values: torch.Tensor, 
                      noise_stddev: torch.Tensor, 
                      noisy_top_values: torch.Tensor) -> torch.Tensor:
        
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.gather(top_values_flat, 0, threshold_positions_if_in)
        
        is_in = torch.gt(noisy_values, threshold_if_in.unsqueeze(1))
        threshold_if_out = torch.gather(top_values_flat, 0, threshold_positions_if_in - 1)
        
        
        prob_if_in = self._normal_cdf((clean_values - threshold_if_in.unsqueeze(1)) / noise_stddev)
        prob_if_out = self._normal_cdf((clean_values - threshold_if_out.unsqueeze(1)) / noise_stddev)
        
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def _normal_cdf(self, x: torch.Tensor) -> torch.Tensor:
        
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
    def forward(self, x: torch.Tensor, train: bool = True, loss_coef: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
        
        clean_logits = self.w_gate(x)
        
        if train:
            
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits
            noise_stddev = torch.zeros_like(clean_logits)
        
        
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = F.softmax(top_k_logits, dim=1)
        
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        if train:
            
            load = self._gates_to_load(gates).float()
            importance = gates.sum(0)
            
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef
        else:
            loss = torch.tensor(0.0, device=x.device)
        
        return gates, loss

class MoELayer(BaseModel):
    
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int, 
                 output_dim: int,
                 num_experts: int = 8,
                 top_k: int = 2,
                 dropout: float = 0.1,
                 noise_epsilon: float = 1e-2):
        super().__init__("MoELayer")
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim
        
        
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        
        self.gate = NoisyTopKGating(input_dim, num_experts, top_k, noise_epsilon)
        
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = x.size(0)
        
        
        gates, aux_loss = self.gate(x, train)
        
        
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=2)  
        
        
        gates = gates.unsqueeze(1)  
        output = torch.bmm(expert_outputs, gates.transpose(1, 2)).squeeze(2)
        
        return output, aux_loss

class SparseMoETransformer(BaseModel):
    
    
    def __init__(self, 
                 input_dim: int = 600,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_experts: int = 8,
                 top_k: int = 2,
                 output_dim: int = 1,
                 dropout: float = 0.1):
        super().__init__("SparseMoETransformer")
        
        self.d_model = d_model
        
        
        self.input_projection = nn.Linear(1, d_model)
        
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True
                )
            )
            
            
            if i % 2 == 1:  
                self.layers.append(
                    MoELayer(
                        input_dim=d_model,
                        hidden_dim=d_model * 2,
                        output_dim=d_model,
                        num_experts=num_experts,
                        top_k=top_k,
                        dropout=dropout
                    )
                )
        
        
        self.output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len = x.shape
        
        
        x = x.unsqueeze(-1)  
        x = self.input_projection(x)  
        
        total_aux_loss = 0.0
        
        
        for layer in self.layers:
            if isinstance(layer, MoELayer):
                
                batch_size, seq_len, d_model = x.shape
                x_flat = x.view(-1, d_model)  
                
                x_out, aux_loss = layer(x_flat, train)
                x = x_out.view(batch_size, seq_len, d_model)
                total_aux_loss += aux_loss
            else:
                
                x = layer(x)
        
        
        x = x.transpose(1, 2)  
        output = self.output_head(x)
        
        return output, total_aux_loss


def create_moe_model():
    
    model = SparseMoETransformer(
        input_dim=600,
        d_model=256,
        num_heads=8,
        num_layers=6,
        num_experts=8,
        top_k=2,
        output_dim=1,
        dropout=0.1
    )
    
    model.print_model_info()
    return model

if __name__ == "__main__":
    
    model = create_moe_model()
    
    
    batch_size, seq_len = 32, 600
    x = torch.randn(batch_size, seq_len)
    
    
    with torch.no_grad():
        output, aux_loss = model(x, train=False)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Aux loss: {aux_loss.item()}")
