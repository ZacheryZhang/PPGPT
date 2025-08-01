


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import logging

from data.data_generator import PPGDataProcessor, DataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Preprocess PPG data')
    parser.add_argument('--input_path', required=True, help='Raw data path')
    parser.add_argument('--output_dir', default='./processed_data', help='Output directory')
    parser.add_argument('--format', choices=['csv', 'npy', 'hdf5'], default='csv', help='Input data format')
    parser.add_argument('--signal_length', type=int, default=600, help='Target signal length')
    parser.add_argument('--image_size', type=int, default=600, help='Target image size')
    parser.add_argument('--fs', type=int, default=64, help='Sampling frequency')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--quality_check', action='store_true', help='Enable signal quality check')
    parser.add_argument('--dataset_name', default='ppg_dataset', help='Dataset name')
    
    args = parser.parse_args()
    
    
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    
    processor = PPGDataProcessor(
        signal_length=args.signal_length,
        image_size=args.image_size,
        fs=args.fs,
        quality_check_enabled=args.quality_check
    )
    
    
    generator = DataGenerator(
        processor=processor,
        output_dir=args.output_dir
    )
    
    logger.info(f"Processing data from {args.input_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Signal length: {args.signal_length}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Sampling frequency: {args.fs}")
    logger.info(f"Data splits: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    
    
    if Path(args.input_path).exists():
        generator.generate_dataset(
            raw_data_path=args.input_path,
            dataset_name=args.dataset_name,
            format_type=args.format,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            n_workers=args.n_workers
        )
    else:
        logger.warning(f"Input path {args.input_path} does not exist. Generating synthetic data for testing...")
        generator.generate_synthetic_data(n_samples=1000)
    
    logger.info("Data preprocessing completed!")

if __name__ == '__main__':
    main()
