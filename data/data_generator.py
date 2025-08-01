import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import os
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
from tqdm import tqdm

from .preprocessing import (
    generate_spatial_features, 
    preprocess_ppg_signal, 
    quality_check
)
from ..utils.feature_extraction import (
    extract_time_features,
    extract_frequency_features,
    extract_morphological_features
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPGDataProcessor:
    
    
    def __init__(self, 
                 signal_length: int = 600,
                 image_size: int = 600,
                 fs: int = 64,
                 quality_check_enabled: bool = True):
        
        self.signal_length = signal_length
        self.image_size = image_size
        self.fs = fs
        self.quality_check_enabled = quality_check_enabled
        
    def process_single_signal(self, 
                            signal: np.ndarray, 
                            label: Optional[np.ndarray] = None) -> Optional[Dict]:
        
        try:
            
            processed_signal = preprocess_ppg_signal(
                signal, 
                target_length=self.signal_length,
                normalize=True
            )
            
            
            if self.quality_check_enabled and not quality_check(processed_signal):
                return None
            
            
            spatial_features = generate_spatial_features(processed_signal)
            
            
            time_features = extract_time_features(processed_signal)
            freq_features = extract_frequency_features(processed_signal, self.fs)
            morph_features = extract_morphological_features(processed_signal)
            
            
            combined_features = np.concatenate([
                time_features, 
                freq_features, 
                morph_features
            ])
            
            result = {
                'signal': processed_signal,
                'spatial_features': spatial_features,
                'combined_features': combined_features,
                'time_features': time_features,
                'freq_features': freq_features,
                'morph_features': morph_features
            }
            
            if label is not None:
                result['label'] = label
                
            return result
            
        except Exception as e:
            logger.warning(f"Error processing signal: {e}")
            return None
    
    def process_batch(self, 
                     signals: np.ndarray, 
                     labels: Optional[np.ndarray] = None,
                     n_workers: int = 4) -> Dict[str, np.ndarray]:
        
        
        results = {
            'signals': [],
            'spatial_features': [],
            'combined_features': [],
            'time_features': [],
            'freq_features': [],
            'morph_features': []
        }
        
        if labels is not None:
            results['labels'] = []
        
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            
            for i in range(len(signals)):
                label = labels[i] if labels is not None else None
                future = executor.submit(self.process_single_signal, signals[i], label)
                futures.append(future)
            
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing signals"):
                result = future.result()
                if result is not None:
                    results['signals'].append(result['signal'])
                    results['spatial_features'].append(result['spatial_features'])
                    results['combined_features'].append(result['combined_features'])
                    results['time_features'].append(result['time_features'])
                    results['freq_features'].append(result['freq_features'])
                    results['morph_features'].append(result['morph_features'])
                    
                    if 'label' in result:
                        results['labels'].append(result['label'])
        
        
        for key, value in results.items():
            if value:
                results[key] = np.array(value)
        
        return results

class DataGenerator:
    
    
    def __init__(self, 
                 processor: PPGDataProcessor,
                 output_dir: str = './processed_data'):
        
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self, 
                     data_path: str, 
                     format_type: str = 'csv') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        if format_type == 'csv':
            df = pd.read_csv(data_path)
            
            if 'SBP' in df.columns or 'DBP' in df.columns:
                label_cols = ['SBP', 'DBP', 'HR', 'GLU']
                label_cols = [col for col in label_cols if col in df.columns]
                labels = df[label_cols].values
                signals = df.drop(columns=label_cols).values
            else:
                signals = df.values
                labels = None
                
        elif format_type == 'npy':
            data = np.load(data_path, allow_pickle=True)
            if isinstance(data, dict):
                signals = data['signals']
                labels = data.get('labels', None)
            else:
                signals = data
                labels = None
                
        elif format_type == 'hdf5':
            with h5py.File(data_path, 'r') as f:
                signals = f['signals'][:]
                labels = f['labels'][:] if 'labels' in f else None
                
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        return signals, labels
    
    def generate_dataset(self, 
                        raw_data_path: str,
                        dataset_name: str = 'dataset',
                        format_type: str = 'csv',
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        n_workers: int = 4) -> None:
        
        logger.info(f"Loading raw data from {raw_data_path}")
        signals, labels = self.load_raw_data(raw_data_path, format_type)
        
        
        n_samples = len(signals)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        splits = {
            'train': (train_indices, 'train'),
            'val': (val_indices, 'val'), 
            'test': (test_indices, 'test')
        }
        
        
        for split_name, (split_indices, prefix) in splits.items():
            logger.info(f"Processing {split_name} set ({len(split_indices)} samples)")
            
            split_signals = signals[split_indices]
            split_labels = labels[split_indices] if labels is not None else None
            
            
            results = self.processor.process_batch(
                split_signals, 
                split_labels,
                n_workers=n_workers
            )
            
            
            self._save_split_data(results, dataset_name, prefix)
    
    def _save_split_data(self, 
                        results: Dict[str, np.ndarray], 
                        dataset_name: str, 
                        prefix: str) -> None:
        
        
        
        save_items = [
            ('signals', 'signals'),
            ('spatial_features', 'images'),  
            ('combined_features', 'features'),
            ('time_features', 'time_features'),
            ('freq_features', 'freq_features'),
            ('morph_features', 'morph_features')
        ]
        
        if 'labels' in results:
            save_items.append(('labels', 'labels'))
        
        for key, filename in save_items:
            if key in results and len(results[key]) > 0:
                save_path = self.output_dir / f"{prefix}_{filename}.npy"
                np.save(save_path, results[key])
                logger.info(f"Saved {key} to {save_path} (shape: {results[key].shape})")
        
        
        metadata = {
            'dataset_name': dataset_name,
            'split': prefix,
            'n_samples': len(results['signals']) if 'signals' in results else 0,
            'signal_length': self.processor.signal_length,
            'image_size': self.processor.image_size,
            'sampling_rate': self.processor.fs
        }
        
        if 'combined_features' in results and len(results['combined_features']) > 0:
            metadata['feature_dim'] = results['combined_features'].shape[1]
        
        metadata_path = self.output_dir / f"{prefix}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def generate_synthetic_data(self, 
                              n_samples: int = 1000,
                              noise_level: float = 0.1) -> None:
        
        logger.info(f"Generating {n_samples} synthetic samples")
        
        
        t = np.linspace(0, 10, self.processor.signal_length)
        signals = []
        labels = []
        
        for i in tqdm(range(n_samples), desc="Generating synthetic data"):
            
            hr = np.random.uniform(60, 120)
            freq = hr / 60  
            
            
            cardiac = np.sin(2 * np.pi * freq * t)
            respiratory = 0.3 * np.sin(2 * np.pi * 0.25 * t)  
            noise = noise_level * np.random.randn(len(t))
            
            signal = cardiac + respiratory + noise
            signals.append(signal)
            
            
            sbp = np.random.uniform(90, 180)  
            dbp = np.random.uniform(60, 120)  
            glu = np.random.uniform(70, 200)  
            
            labels.append([sbp, dbp, hr, glu])
        
        signals = np.array(signals)
        labels = np.array(labels)
        
        
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        train_signals, train_labels = signals[:n_train], labels[:n_train]
        val_signals, val_labels = signals[n_train:n_train+n_val], labels[n_train:n_train+n_val]
        test_signals, test_labels = signals[n_train+n_val:], labels[n_train+n_val:]
        
        
        splits = [
            (train_signals, train_labels, 'train'),
            (val_signals, val_labels, 'val'),
            (test_signals, test_labels, 'test')
        ]
        
        for split_signals, split_labels, prefix in splits:
            results = self.processor.process_batch(split_signals, split_labels)
            self._save_split_data(results, 'synthetic', prefix)
        
        logger.info("Synthetic data generation completed")


def create_sample_dataset():
    
    
    
    processor = PPGDataProcessor(
        signal_length=600,
        image_size=600,
        fs=64,
        quality_check_enabled=True
    )
    
    
    generator = DataGenerator(
        processor=processor,
        output_dir='./processed_data'
    )
    
    
    generator.generate_synthetic_data(n_samples=1000)
    
    print("Sample dataset created successfully!")
    print("Files generated:")
    print("- train_signals.npy, train_images.npy, train_labels.npy")
    print("- val_signals.npy, val_images.npy, val_labels.npy") 
    print("- test_signals.npy, test_images.npy, test_labels.npy")

if __name__ == "__main__":
    create_sample_dataset()
