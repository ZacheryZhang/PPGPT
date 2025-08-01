import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import torch

def plot_training_history(history: Dict[str, List], save_path: Optional[str] = None):
    
    fig, axes = plt.subplots(1, len(history), figsize=(6*len(history), 5))
    if len(history) == 1:
        axes = [axes]
    
    for i, (metric_name, values) in enumerate(history.items()):
        axes[i].plot(values, label=metric_name)
        axes[i].set_title(f'{metric_name.replace("_", " ").title()} Over Epochs')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric_name.replace("_", " ").title())
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions_vs_actual(y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              title: str = "Predictions vs Actual",
                              save_path: Optional[str] = None):
    
    plt.figure(figsize=(8, 8))
    
    
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    
    from utils.metrics import calculate_metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    
    textstr = f'MAE: {metrics["mae"]:.3f}\nRMSE: {metrics["rmse"]:.3f}\nR²: {metrics["r2"]:.3f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_multi_task_results(results: Dict[str, Dict[str, np.ndarray]], 
                           save_path: Optional[str] = None):
    
    task_names = list(results.keys())
    n_tasks = len(task_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, task_name in enumerate(task_names):
        if i < 4:  
            y_true = results[task_name]['true']
            y_pred = results[task_name]['pred']
            
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=20)
            
            
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            axes[i].set_xlabel('Actual Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].set_title(f'{task_name} Predictions')
            axes[i].grid(True, alpha=0.3)
            
            
            from utils.metrics import calculate_metrics
            metrics = calculate_metrics(y_true, y_pred)
            textstr = f'R²: {metrics["r2"]:.3f}\nRMSE: {metrics["rmse"]:.3f}'
            axes[i].text(0.05, 0.95, textstr, transform=axes[i].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    
    for i in range(n_tasks, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model: torch.nn.Module, 
                          feature_names: List[str],
                          save_path: Optional[str] = None):
    
    
    importances = []
    
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            importance = torch.abs(param).mean().item()
            importances.append(importance)
    
    if len(importances) == len(feature_names):
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        plt.barh(df['feature'], df['importance'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Analysis')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def plot_signal_and_image(signal: np.ndarray, 
                         image: np.ndarray, 
                         title: str = "PPG Signal and Spatial Features",
                         save_path: Optional[str] = None):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    
    axes[0, 0].plot(signal)
    axes[0, 0].set_title('PPG Signal')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    
    channel_names = ['GASF', 'MTF', 'TDM']
    
    for i in range(3):
        row = i // 2 + (1 if i >= 1 else 0)
        col = i % 2 if i < 2 else 1
        
        if i == 0:
            row, col = 0, 1
        elif i == 1:
            row, col = 1, 0
        else:
            row, col = 1, 1
            
        im = axes[row, col].imshow(image[:, :, i], cmap='viridis')
        axes[row, col].set_title(f'{channel_names[i]} Channel')
        plt.colorbar(im, ax=axes[row, col])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
