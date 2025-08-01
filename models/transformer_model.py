import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .base_model import EncoderModel

class PositionalEncoding(nn.Module):
    
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  
            scores = scores.masked_fill(mask == 0, -1e9)
        
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        
        context = torch.matmul(attn_weights, V)
        
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(context)
        return output

class FeedForwardNetwork(nn.Module):
    
    
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    
    
    def __init__(self, 
                 num_layers: int, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 max_seq_len: int = 1000,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        seq_len = x.size(1)
        
        
        x = x * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  
        
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class PPGTransformer(EncoderModel):
    
    
    def __init__(self, 
                 input_dim: int = 600,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 1024,
                 output_dim: int = 256,
                 max_seq_len: int = 1000,
                 dropout: float = 0.1,
                 pooling: str = 'mean'):
        
        super().__init__(input_dim, output_dim, "PPGTransformer")
        
        self.d_model = d_model
        self.pooling = pooling
        
        
        self.input_projection = nn.Linear(1, d_model)
        
        
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.initialize_weights()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        
        batch_size, seq_len = x.shape
        
        
        x = x.unsqueeze(-1)  
        
        
        x = self.input_projection(x)  
        
        
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        
        encoded = self.transformer(x)  
        
        
        if self.pooling == 'mean':
            pooled = torch.mean(encoded, dim=1)
        elif self.pooling == 'max':
            pooled, _ = torch.max(encoded, dim=1)
        elif self.pooling == 'cls':
            pooled = encoded[:, 0]  
        elif self.pooling == 'last':
            pooled = encoded[:, -1]  
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        
        
        output = self.output_projection(pooled)
        
        return output

class PPGTransformerClassifier(PPGTransformer):
    
    
    def __init__(self, 
                 num_classes: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = "PPGTransformerClassifier"
        
        
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        logits = self.classifier(features)
        return logits


def create_ppg_transformer():
    
    model = PPGTransformer(
        input_dim=600,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        output_dim=256,
        dropout=0.1,
        pooling='mean'
    )
    
    model.print_model_info()
    return model

if __name__ == "__main__":
    
    model = create_ppg_transformer()
    
    
    batch_size, seq_len = 32, 600
    x = torch.randn(batch_size, seq_len)
    
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
