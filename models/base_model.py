import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(nn.Module, ABC):
    
    
    def __init__(self, model_name: str = "BaseModel"):
        super().__init__()
        self.model_name = model_name
        self._initialized = False
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        
        pass
    
    def initialize_weights(self):
        
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        self._initialized = True
        logger.info(f"{self.model_name} weights initialized")
    
    def get_model_info(self) -> Dict[str, Any]:
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  
            'initialized': self._initialized
        }
    
    def print_model_info(self):
        
        info = self.get_model_info()
        print(f"\n{'='*50}")
        print(f"Model: {info['model_name']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Model Size: {info['model_size_mb']:.2f} MB")
        print(f"Initialized: {info['initialized']}")
        print(f"{'='*50}\n")
    
    def freeze_layers(self, layer_names: Optional[list] = None):
        
        if layer_names is None:
            
            for param in self.parameters():
                param.requires_grad = False
        else:
            
            for name, module in self.named_modules():
                if name in layer_names:
                    for param in module.parameters():
                        param.requires_grad = False
        
        logger.info(f"Frozen layers: {layer_names if layer_names else 'all'}")
    
    def unfreeze_layers(self, layer_names: Optional[list] = None):
        
        if layer_names is None:
            
            for param in self.parameters():
                param.requires_grad = True
        else:
            
            for name, module in self.named_modules():
                if name in layer_names:
                    for param in module.parameters():
                        param.requires_grad = True
        
        logger.info(f"Unfrozen layers: {layer_names if layer_names else 'all'}")
    
    def save_model(self, path: str, save_full_model: bool = False):
        
        if save_full_model:
            torch.save(self, path)
        else:
            torch.save({
                'model_state_dict': self.state_dict(),
                'model_info': self.get_model_info()
            }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str, device: str = 'cpu'):
        
        checkpoint = torch.load(path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
            if 'model_info' in checkpoint:
                logger.info(f"Loaded model info: {checkpoint['model_info']}")
        else:
            
            self.load_state_dict(checkpoint)
        
        logger.info(f"Model loaded from {path}")
    
    def count_parameters(self) -> Tuple[int, int]:
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

class EncoderModel(BaseModel):
    
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 model_name: str = "EncoderModel"):
        super().__init__(model_name)
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.encode(x)

class PredictorModel(BaseModel):
    
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int = 1,
                 model_name: str = "PredictorModel"):
        super().__init__(model_name)
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.predict(x)

class MultiModalModel(BaseModel):
    
    
    def __init__(self, 
                 model_name: str = "MultiModalModel"):
        super().__init__(model_name)
    
    @abstractmethod
    def encode_modalities(self, *modalities) -> Dict[str, torch.Tensor]:
        
        pass
    
    @abstractmethod
    def fuse_modalities(self, encoded_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        pass
    
    def forward(self, *modalities) -> torch.Tensor:
        
        encoded = self.encode_modalities(*modalities)
        fused = self.fuse_modalities(encoded)
        return fused
