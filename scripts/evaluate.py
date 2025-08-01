


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from pathlib import Path
import yaml
import json

from data.dataset import PPGDataset, MultiTaskDataset
from models.dual_stream_encoder import DualStreamEncoder
from models.expert_model import MultiExpertModel
from models.transformer_model import PPGTransformer
from utils.metrics import calculate_metrics, MetricsTracker
from utils.visualization import plot_predictions_vs_actual, plot_multi_task_results
from torch.utils.data import DataLoader

def load_model(model_path: str, config: dict):
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model_type = config['model']['type']
    model_config = config['model']['config']
    
    if model_type == 'dual_stream':
        model = DualStreamEncoder(**model_config)
    elif model_type == 'multi_expert':
        model = MultiExpertModel(**model_config)
    elif model_type == 'transformer':
        model = PPGTransformer(**model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate_single_task(model, test_loader, device, task_name=''):
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                data, _, labels = batch
                data, labels = data.to(device), labels.to(device)
                
                predictions = model(data)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
            else:
                
                data = batch[0].to(device)
                predictions = model(data)
                all_predictions.extend(predictions.cpu().numpy().flatten())
    
    if all_labels:
        
        metrics = calculate_metrics(all_labels, all_predictions)
        
        print(f"\n{task_name} Evaluation Results:")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAPE: {metrics['mape']:.4f}%")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"Pearson: {metrics['pearson']:.4f}")
        
        return {
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'metrics': metrics
        }
    else:
        return {
            'predictions': np.array(all_predictions)
        }

def evaluate_multi_task(model, test_loader, device, expert_names):
    
    model.eval()
    model.to(device)
    
    results = {name: {'predictions': [], 'labels': []} for name in expert_names}
    
    with torch.no_grad():
        for features, labels, masks in test_loader:
            features = features.to(device)
            outputs = model(features)
            
            for i, expert_name in enumerate(expert_names):
                task_mask = masks[:, i].bool()
                if task_mask.sum() > 0:
                    pred = outputs[expert_name][task_mask].cpu().numpy().flatten()
                    true = labels[:, i][task_mask].numpy().flatten()
                    
                    results[expert_name]['predictions'].extend(pred)
                    results[expert_name]['labels'].extend(true)
    
    
    task_metrics = {}
    for expert_name in expert_names:
        if len(results[expert_name]['predictions']) > 0:
            preds = np.array(results[expert_name]['predictions'])
            labels = np.array(results[expert_name]['labels'])
            
            metrics = calculate_metrics(labels, preds)
            task_metrics[expert_name] = metrics
            
            print(f"\n{expert_name} Evaluation Results:")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAPE: {metrics['mape']:.4f}%")
            print(f"R²: {metrics['r2']:.4f}")
            print(f"Pearson: {metrics['pearson']:.4f}")
            
            results[expert_name]['predictions'] = preds
            results[expert_name]['labels'] = labels
            results[expert_name]['metrics'] = metrics
    
    return results, task_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate PPG models')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--model_path', required=True, help='Model checkpoint path')
    parser.add_argument('--data_dir', required=True, help='Test data directory')
    parser.add_argument('--output_dir', default='./eval_outputs', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    
    args = parser.parse_args()
    
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    device = torch.device(config['training']['device'])
    
    
    print("Loading model...")
    model = load_model(args.model_path, config)
    model.print_model_info()
    
    
    data_dir = Path(args.data_dir)
    model_type = config['model']['type']
    
    if model_type == 'dual_stream':
        test_dataset = PPGDataset(
            signal_path=data_dir / 'test_signals.npy',
            image_path=data_dir / 'test_images.npy',
            label_path=data_dir / 'test_labels.npy' if (data_dir / 'test_labels.npy').exists() else None
        )
    else:
        test_dataset = MultiTaskDataset(
            features_path=data_dir / 'test_features.npy',
            labels_path=data_dir / 'test_labels.npy'
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    
    if model_type == 'multi_expert':
        expert_names = config['model']['config'].get('expert_names', ['SBP', 'DBP', 'HR', 'GLU'])
        results, task_metrics = evaluate_multi_task(model, test_loader, device, expert_names)
        
        
        results_path = output_dir / 'multi_task_results.json'
        with open(results_path, 'w') as f:
            
            json_results = {}
            for task, data in results.items():
                json_results[task] = {
                    'predictions': data['predictions'].tolist() if 'predictions' in data else [],
                    'labels': data['labels'].tolist() if 'labels' in data else [],
                    'metrics': data.get('metrics', {})
                }
            json.dump(json_results, f, indent=2)
        
        
        if args.visualize:
            viz_results = {}
            for task, data in results.items():
                if len(data['predictions']) > 0:
                    viz_results[task] = {
                        'pred': data['predictions'],
                        'true': data['labels']
                    }
            
            if viz_results:
                plot_multi_task_results(
                    viz_results,
                    save_path=output_dir / 'multi_task_predictions.png'
                )
        
    else:
        
        result = evaluate_single_task(model, test_loader, device, 'Single Task')
        
        if 'labels' in result:
            
            results_path = output_dir / 'single_task_results.json'
            with open(results_path, 'w') as f:
                json_result = {
                    'predictions': result['predictions'].tolist(),
                    'labels': result['labels'].tolist(),
                    'metrics': result['metrics']
                }
                json.dump(json_result, f, indent=2)
            
            
            if args.visualize:
                plot_predictions_vs_actual(
                    result['labels'],
                    result['predictions'],
                    title='Model Predictions vs Actual Values',
                    save_path=output_dir / 'predictions_vs_actual.png'
                )
    
    print(f"\nEvaluation completed! Results saved to {output_dir}")

if __name__ == '__main__':
    main()
