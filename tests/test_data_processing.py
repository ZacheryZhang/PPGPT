


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from data.preprocessing import (
    generate_spatial_features, 
    preprocess_ppg_signal, 
    quality_check,
    compute_tdm
)
from data.data_generator import PPGDataProcessor, DataGenerator
from utils.feature_extraction import (
    extract_time_features,
    extract_frequency_features,
    extract_morphological_features
)

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        
        
        t = np.linspace(0, 10, 600)
        self.test_signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(600)
        self.signal_length = 600
    
    def test_preprocess_ppg_signal(self):
        
        
        processed = preprocess_ppg_signal(self.test_signal, self.signal_length)
        self.assertEqual(len(processed), self.signal_length)
        self.assertEqual(processed.dtype, np.float32)
        
        
        short_signal = self.test_signal[:300]
        processed_short = preprocess_ppg_signal(short_signal, self.signal_length)
        self.assertEqual(len(processed_short), self.signal_length)
        
        long_signal = np.concatenate([self.test_signal, self.test_signal])
        processed_long = preprocess_ppg_signal(long_signal, self.signal_length)
        self.assertEqual(len(processed_long), self.signal_length)
    
    def test_quality_check(self):
        
        
        self.assertTrue(quality_check(self.test_signal))
        
        
        flat_signal = np.ones(600) * 0.5
        self.assertFalse(quality_check(flat_signal))
        
        
        noisy_signal = np.random.randn(600) * 10
        self.assertFalse(quality_check(noisy_signal))
    
    def test_generate_spatial_features(self):
        
        spatial_features = generate_spatial_features(self.test_signal)
        
        
        expected_shape = (len(self.test_signal), len(self.test_signal), 3)
        self.assertEqual(spatial_features.shape, expected_shape)
        
        
        self.assertFalse(np.isnan(spatial_features).any())
        self.assertFalse(np.isinf(spatial_features).any())
    
    def test_compute_tdm(self):
        
        tdm = compute_tdm(self.test_signal)
        
        
        expected_shape = (len(self.test_signal), len(self.test_signal))
        self.assertEqual(tdm.shape, expected_shape)
        
        
        np.testing.assert_array_almost_equal(np.diag(tdm), 0)

class TestFeatureExtraction(unittest.TestCase):
    
    def setUp(self):
        
        t = np.linspace(0, 10, 600)
        self.test_signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(600)
    
    def test_extract_time_features(self):
        
        features = extract_time_features(self.test_signal)
        
        
        self.assertGreater(len(features), 0)
        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isinf(features).any())
    
    def test_extract_frequency_features(self):
        
        features = extract_frequency_features(self.test_signal, fs=64)
        
        
        self.assertGreater(len(features), 0)
        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isinf(features).any())
    
    def test_extract_morphological_features(self):
        
        features = extract_morphological_features(self.test_signal)
        
        
        self.assertGreater(len(features), 0)
        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isinf(features).any())

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        
        self.processor = PPGDataProcessor(
            signal_length=600,
            image_size=600,
            fs=64,
            quality_check_enabled=False  
        )
        
        
        t = np.linspace(0, 10, 600)
        self.test_signals = []
        self.test_labels = []
        
        for i in range(10):
            signal = np.sin(2 * np.pi * (1 + i * 0.1) * t) + 0.1 * np.random.randn(600)
            label = [120 + i, 80 + i, 70 + i, 100 + i]  
            
            self.test_signals.append(signal)
            self.test_labels.append(label)
        
        self.test_signals = np.array(self.test_signals)
        self.test_labels = np.array(self.test_labels)
    
    def test_process_single_signal(self):
        
        result = self.processor.process_single_signal(
            self.test_signals[0], 
            self.test_labels[0]
        )
        
        self.assertIsNotNone(result)
        self.assertIn('signal', result)
        self.assertIn('spatial_features', result)
        self.assertIn('combined_features', result)
        self.assertIn('label', result)
        
        
        self.assertEqual(result['signal'].shape, (600,))
        self.assertEqual(result['spatial_features'].shape, (600, 600, 3))
        self.assertGreater(len(result['combined_features']), 0)
    
    def test_process_batch(self):
        
        results = self.processor.process_batch(
            self.test_signals,
            self.test_labels,
            n_workers=2
        )
        
        self.assertIn('signals', results)
        self.assertIn('spatial_features', results)
        self.assertIn('combined_features', results)
        self.assertIn('labels', results)
        
        
        n_processed = len(results['signals'])
        self.assertGreater(n_processed, 0)
        self.assertEqual(len(results['spatial_features']), n_processed)
        self.assertEqual(len(results['labels']), n_processed)

class TestDataGenerator(unittest.TestCase):
    
    def setUp(self):
        
        self.temp_dir = tempfile.mkdtemp()
        
        processor = PPGDataProcessor(
            signal_length=100,  
            image_size=100,
            fs=64,
            quality_check_enabled=False
        )
        
        self.generator = DataGenerator(
            processor=processor,
            output_dir=self.temp_dir
        )
    
    def tearDown(self):
        
        shutil.rmtree(self.temp_dir)
    
    def test_generate_synthetic_data(self):
        
        self.generator.generate_synthetic_data(n_samples=50)
        
        
        temp_path = Path(self.temp_dir)
        
        expected_files = [
            'train_signals.npy',
            'train_images.npy', 
            'train_labels.npy',
            'val_signals.npy',
            'val_images.npy',
            'val_labels.npy',
            'test_signals.npy',
            'test_images.npy',
            'test_labels.npy'
        ]
        
        for file_name in expected_files:
            file_path = temp_path / file_name
            self.assertTrue(file_path.exists(), f"Missing file: {file_name}")
            
            
            data = np.load(file_path)
            self.assertGreater(len(data), 0)

if __name__ == '__main__':
    unittest.main()
