import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Union
import torch

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    return mean_absolute_error(y_true, y_pred)

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    return r2_score(y_true, y_pred)

def calculate_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    return np.corrcoef(y_true, y_pred)[0, 1]

def calculate_metrics(y_true: Union[np.ndarray, List], 
                     y_pred: Union[np.ndarray, List]) -> Dict[str, float]:
    
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    
    metrics = {
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'pearson': calculate_pearson_correlation(y_true, y_pred)
    }
    
    return metrics

class MetricsTracker:
    
    
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.reset()
    
    def reset(self):
        
        self.metrics = {name: [] for name in self.metric_names}
    
    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor, task_name: str = 'default'):
        
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        
        current_metrics = calculate_metrics(y_true_np, y_pred_np)
        
        for name in self.metric_names:
            if name in current_metrics:
                self.metrics[name].append(current_metrics[name])
    
    def get_average_metrics(self) -> Dict[str, float]:
        
        avg_metrics = {}
        for name, values in self.metrics.items():
            if values:
                avg_metrics[name] = np.mean(values)
            else:
                avg_metrics[name] = 0.0
        return avg_metrics
    
    def print_metrics(self, prefix: str = ""):
        
        avg_metrics = self.get_average_metrics()
        for name, value in avg_metrics.items():
            print(f"{prefix}{name.upper()}: {value:.4f}")
