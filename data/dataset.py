import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
import os

class PPGDataset(Dataset):
    
    
    def __init__(self, 
                 signal_path: str, 
                 image_path: str,
                 label_path: Optional[str] = None,
                 target_idx: Optional[int] = None):
        
        self.signals = np.load(signal_path).astype(np.float32)
        self.images = np.load(image_path).astype(np.float32)
        
        if label_path is not None:
            all_labels = np.load(label_path).astype(np.float32)
            if target_idx is not None:
                self.labels = all_labels[:, target_idx]
            else:
                self.labels = all_labels
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx])
        image = torch.from_numpy(self.images[idx])
        
        if self.labels is not None:
            label = torch.from_numpy(np.array([self.labels[idx]]))
            return signal, image, label
        return signal, image

class MultiTaskDataset(Dataset):
    
    
    def __init__(self, features_path: str, labels_path: str):
        self.features = np.load(features_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.float32)
        
        
        self.valid_mask = self.labels != -1
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx])
        label = torch.from_numpy(self.labels[idx])
        mask = torch.from_numpy(self.valid_mask[idx].astype(np.float32))
        return feature, label, mask
