import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalEncoder(nn.Module):
    
    
    def __init__(self, input_dim: int = 600, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        
        conv_output_size = input_dim // 8 * 128
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        
        x = x.unsqueeze(1)  
        x = self.conv_layers(x)  
        x = x.flatten(1)  
        x = self.fc_layers(x)  
        return x

class ImageEncoder(nn.Module):
    
    
    def __init__(self, input_channels: int = 3, output_dim: int = 256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

class CrossAttentionFusion(nn.Module):
    
    
    def __init__(self, feature_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, signal_feat, image_feat):
        
        signal_feat = signal_feat.unsqueeze(1)  
        image_feat = image_feat.unsqueeze(1)    
        
        
        fused_feat, _ = self.multihead_attn(
            query=signal_feat,
            key=image_feat, 
            value=image_feat
        )
        
        
        fused_feat = self.norm(fused_feat + signal_feat)
        return fused_feat.squeeze(1)  

class DualStreamEncoder(nn.Module):
    
    
    def __init__(self, 
                 signal_dim: int = 600,
                 image_channels: int = 3,
                 output_dim: int = 256):
        super().__init__()
        
        self.signal_encoder = SignalEncoder(signal_dim, output_dim, output_dim)
        self.image_encoder = ImageEncoder(image_channels, output_dim)
        self.fusion = CrossAttentionFusion(output_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, signal, image):
        signal_feat = self.signal_encoder(signal)
        image_feat = self.image_encoder(image)
        
        fused_feat = self.fusion(signal_feat, image_feat)
        output = self.projection(fused_feat)
        
        return output, signal_feat, image_feat
