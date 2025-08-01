from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class ModelConfig:
    
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 16
    max_seq_len: int = 600
    dropout: float = 0.1
    
    
    n_routed_experts: int = 64
    n_activated_experts: int = 6
    n_shared_experts: int = 2
    n_expert_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    
    
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    epochs: int = 100
    patience: int = 20
    
    
    signal_length: int = 600
    image_size: int = 600
    num_channels: int = 3

@dataclass
class ExpertConfig:
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    patience: int = 100
