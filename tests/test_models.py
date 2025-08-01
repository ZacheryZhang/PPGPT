


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import numpy as np

from models.dual_stream_encoder import DualStreamEncoder, SignalEncoder, ImageEncoder
from models.expert_model import MultiExpertModel, ExpertNetwork
from models.transformer_model import PPGTransformer
from models.base_model import BaseModel

class TestModels(unittest.TestCase):
    
    def setUp(self):
        
        self.batch_size = 8
        self.signal_length = 600
        self.image_size = 600
        self.num_channels = 3
        self.feature_dim = 256
        
        
        self.test_signal = torch.randn(self.batch_size, self.signal_length)
        self.test_image = torch.randn(self.batch_size, self.image_size, self.image_size, self.num_channels)
        self.test_features = torch.randn(self.batch_size, self.feature_dim)
        self.test_labels = torch.randn(self.batch_size, 4)
        self.test_mask = torch.ones(self.batch_size, 4)
    
    def test_signal_encoder(self):
        
        encoder = SignalEncoder(
            input_dim=self.signal_length,
            hidden_dim=256,
            output_dim=self.feature_dim
        )
        
        output = encoder(self.test_signal)
        self.assertEqual(output.shape, (self.batch_size, self.feature_dim))
        
        
        info = encoder.get_model_info()
        self.assertIn('total_parameters', info)
        self.assertIn('trainable_parameters', info)
    
    def test_image_encoder(self):
        
        encoder = ImageEncoder(
            input_channels=self.num_channels,
            output_dim=self.feature_dim
        )
        
        output = encoder(self.test_image)
        self.assertEqual(output.shape, (self.batch_size, self.feature_dim))
    
    def test_dual_stream_encoder(self):
        
        encoder = DualStreamEncoder(
            signal_dim=self.signal_length,
            image_channels=self.num_channels,
            output_dim=self.feature_dim
        )
        
        output, signal_feat, image_feat = encoder(self.test_signal, self.test_image)
        
        self.assertEqual(output.shape, (self.batch_size, self.feature_dim))
        self.assertEqual(signal_feat.shape, (self.batch_size, self.feature_dim))
        self.assertEqual(image_feat.shape, (self.batch_size, self.feature_dim))
    
    def test_expert_network(self):
        
        expert = ExpertNetwork(
            input_dim=self.feature_dim,
            hidden_dim=512,
            output_dim=1,
            num_layers=3
        )
        
        output = expert(self.test_features)
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_multi_expert_model(self):
        
        model = MultiExpertModel(
            input_dim=self.feature_dim,
            hidden_dim=512,
            expert_names=['SBP', 'DBP', 'HR', 'GLU']
        )
        
        
        outputs = model(self.test_features)
        self.assertEqual(len(outputs), 4)
        for name in ['SBP', 'DBP', 'HR', 'GLU']:
            self.assertIn(name, outputs)
            self.assertEqual(outputs[name].shape, (self.batch_size, 1))
        
        
        single_output = model(self.test_features, target_expert='SBP')
        self.assertEqual(single_output.shape, (self.batch_size, 1))
    
    def test_transformer_model(self):
        
        model = PPGTransformer(
            input_dim=self.signal_length,
            d_model=256,
            num_heads=8,
            num_layers=4,
            output_dim=self.feature_dim
        )
        
        output = model(self.test_signal)
        self.assertEqual(output.shape, (self.batch_size, self.feature_dim))
    
    def test_model_save_load(self):
        
        model = MultiExpertModel(
            input_dim=self.feature_dim,
            hidden_dim=256,
            expert_names=['SBP', 'DBP']
        )
        
        
        model.save_model('test_model.pth')
        
        
        new_model = MultiExpertModel(
            input_dim=self.feature_dim,
            hidden_dim=256,
            expert_names=['SBP', 'DBP']
        )
        new_model.load_model('test_model.pth')
        
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            original_output = model(self.test_features)
            loaded_output = new_model(self.test_features)
            
            for name in ['SBP', 'DBP']:
                torch.testing.assert_close(
                    original_output[name], 
                    loaded_output[name],
                    rtol=1e-5,
                    atol=1e-5
                )
        
        
        os.remove('test_model.pth')

class TestDataConsistency(unittest.TestCase):
    
    
    def test_data_shapes(self):
        
        batch_size = 16
        signal_length = 600
        
        
        signals = torch.randn(batch_size, signal_length)
        images = torch.randn(batch_size, signal_length, signal_length, 3)
        
        
        encoder = DualStreamEncoder(
            signal_dim=signal_length,
            image_channels=3,
            output_dim=256
        )
        
        
        output, _, _ = encoder(signals, images)
        
        
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 256)

if __name__ == '__main__':
    unittest.main()
