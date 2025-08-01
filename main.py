import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os

from config.model_config import ModelConfig, ExpertConfig
from data.dataset import PPGDataset, MultiTaskDataset
from models.dual_stream_encoder import DualStreamEncoder
from models.expert_model import MultiExpertModel
from training.trainer import DualStreamTrainer, MultiExpertTrainer
from utils.metrics import calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='PPG Physiological Parameter Prediction')
    parser.add_argument('--mode', choices=['pretrain', 'expert_train', 'inference'], 
                       required=True, help='Running mode')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--model_dir', type=str, default='./models', help='Model save directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    return parser.parse_args()

def pretrain_dual_stream(args):
    
    print("Starting dual-stream encoder pretraining...")
    
    
    train_dataset = PPGDataset(
        signal_path=os.path.join(args.data_dir, 'train_signals.npy'),
        image_path=os.path.join(args.data_dir, 'train_images.npy')
    )
    val_dataset = PPGDataset(
        signal_path=os.path.join(args.data_dir, 'val_signals.npy'),
        image_path=os.path.join(args.data_dir, 'val_images.npy')
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    
    config = ModelConfig()
    model = DualStreamEncoder(
        signal_dim=config.signal_length,
        image_channels=config.num_channels,
        output_dim=config.dim
    )
    
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    trainer = DualStreamTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device
    )
    
    
    history = trainer.train(args.epochs)
    
    
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'dual_stream_encoder.pth'))
    print("Dual-stream encoder pretraining completed!")

def train_experts(args):
    
    print("Starting multi-expert model training...")
    
    
    train_dataset = MultiTaskDataset(
        features_path=os.path.join(args.data_dir, 'train_features.npy'),
        labels_path=os.path.join(args.data_dir, 'train_labels.npy')
    )
    val_dataset = MultiTaskDataset(
        features_path=os.path.join(args.data_dir, 'val_features.npy'),
        labels_path=os.path.join(args.data_dir, 'val_labels.npy')
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    
    expert_config = ExpertConfig()
    model = MultiExpertModel(
        input_dim=256,
        hidden_dim=expert_config.hidden_dim,
        expert_names=['SBP', 'DBP', 'HR', 'GLU']
    )
    
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=expert_config.weight_decay)
    criterion = nn.MSELoss()
    
    trainer = MultiExpertTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        patience=expert_config.patience
    )
    
    
    history = trainer.train(args.epochs)
    
    
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'multi_expert_model.pth'))
    print("Multi-expert model training completed!")

def inference(args):
    
    print("Starting inference...")
    
    
    test_dataset = MultiTaskDataset(
        features_path=os.path.join(args.data_dir, 'test_features.npy'),
        labels_path=os.path.join(args.data_dir, 'test_labels.npy')
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    
    model = MultiExpertModel(input_dim=256, hidden_dim=512, expert_names=['SBP', 'DBP', 'HR', 'GLU'])
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'multi_expert_model.pth')))
    model.to(args.device)
    model.eval()
    
    
    all_predictions = {name: [] for name in model.expert_names}
    all_labels = {name: [] for name in model.expert_names}
    
    with torch.no_grad():
        for features, labels, masks in test_loader:
            features = features.to(args.device)
            outputs = model(features)
            
            for i, expert_name in enumerate(model.expert_names):
                task_mask = masks[:, i].bool()
                if task_mask.sum() > 0:
                    pred = outputs[expert_name][task_mask].cpu().numpy()
                    true = labels[:, i][task_mask].numpy()
                    
                    all_predictions[expert_name].extend(pred.flatten())
                    all_labels[expert_name].extend(true.flatten())
    
    
    for expert_name in model.expert_names:
        if len(all_predictions[expert_name]) > 0:
            metrics = calculate_metrics(all_labels[expert_name], all_predictions[expert_name])
            print(f"{expert_name} - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
                  f"R²: {metrics['r2']:.4f}")

def main():
    args = parse_args()
    
    if args.mode == 'pretrain':
        pretrain_dual_stream(args)
    elif args.mode == 'expert_train':
        train_experts(args)
    elif args.mode == 'inference':
        inference(args)

if __name__ == '__main__':
    main()