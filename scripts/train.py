


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path
import yaml
import logging

from config.model_config import ModelConfig, ExpertConfig
from data.dataset import PPGDataset, MultiTaskDataset
from data.data_generator import PPGDataProcessor, DataGenerator
from models.dual_stream_encoder import DualStreamEncoder
from models.expert_model import MultiExpertModel
from models.transformer_model import PPGTransformer
from training.trainer import DualStreamTrainer, MultiExpertTrainer
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def setup_data_loaders(config):
    
    data_dir = Path(config['data']['data_dir'])
    
    if config['model']['type'] == 'dual_stream':
        train_dataset = PPGDataset(
            signal_path=data_dir / 'train_signals.npy',
            image_path=data_dir / 'train_images.npy',
            label_path=data_dir / 'train_labels.npy' if 'train_labels.npy' in os.listdir(data_dir) else None
        )
        val_dataset = PPGDataset(
            signal_path=data_dir / 'val_signals.npy',
            image_path=data_dir / 'val_images.npy',
            label_path=data_dir / 'val_labels.npy' if 'val_labels.npy' in os.listdir(data_dir) else None
        )
    else:
        train_dataset = MultiTaskDataset(
            features_path=data_dir / 'train_features.npy',
            labels_path=data_dir / 'train_labels.npy'
        )
        val_dataset = MultiTaskDataset(
            features_path=data_dir / 'val_features.npy',
            labels_path=data_dir / 'val_labels.npy'
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader

def setup_model(config):
    
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
    
    return model

def setup_trainer(model, train_loader, val_loader, config):
    
    
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 0)
        )
    elif optimizer_config['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
    
    
    criterion = torch.nn.MSELoss()
    
    
    model_type = config['model']['type']
    if model_type == 'dual_stream':
        trainer = DualStreamTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config['training']['device'],
            patience=config['training'].get('patience', 20)
        )
    else:
        trainer = MultiExpertTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config['training']['device'],
            patience=config['training'].get('patience', 20)
        )
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description='Train PPG models')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--output_dir', default='./outputs', help='Output directory')
    parser.add_argument('--resume', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    
    config = load_config(args.config)
    
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    device = torch.device(config['training']['device'])
    logger.info(f"Using device: {device}")
    
    
    logger.info("Setting up data loaders...")
    train_loader, val_loader = setup_data_loaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    
    logger.info("Setting up model...")
    model = setup_model(config)
    model.print_model_info()
    
    
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    
    logger.info("Setting up trainer...")
    trainer = setup_trainer(model, train_loader, val_loader, config)
    
    
    logger.info("Starting training...")
    epochs = config['training']['epochs']
    history = trainer.train(epochs - start_epoch)
    
    
    final_model_path = output_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, final_model_path)
    
    logger.info(f"Training completed! Model saved to {final_model_path}")

if __name__ == '__main__':
    main()
