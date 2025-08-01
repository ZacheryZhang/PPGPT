import numpy as np
from scipy.signal import resample
from pyts.image import GramianAngularField, MarkovTransitionField
from typing import Tuple

def compute_tdm(signal: np.ndarray) -> np.ndarray:
    
    return signal[:, np.newaxis] - signal[np.newaxis, :]

def generate_spatial_features(ppg: np.ndarray) -> np.ndarray:
    
    n = ppg.shape[0]
    if np.isnan(ppg).any():
        ppg = np.nan_to_num(ppg, nan=0)
    
    ppg_reshaped = ppg.reshape(1, -1)
    
    
    gasf_transformer = GramianAngularField(method='summation', image_size=n)
    gasf = gasf_transformer.fit_transform(ppg_reshaped)[0]
    
    
    mtf_transformer = MarkovTransitionField(n_bins=8, image_size=n)
    mtf = mtf_transformer.fit_transform(ppg_reshaped)[0]
    
    
    tdm = compute_tdm(ppg)
    
    
    merged_image = np.dstack((gasf, mtf, tdm))
    return merged_image

def preprocess_ppg_signal(signal: np.ndarray, 
                         target_length: int = 600,
                         normalize: bool = True) -> np.ndarray:
    
    
    if len(signal) != target_length:
        signal = resample(signal, target_length)
    
    
    if normalize:
        signal = (signal - np.mean(signal)) / np.std(signal)
    
    return signal.astype(np.float32)

def quality_check(signal: np.ndarray, 
                 min_std: float = 0.01,
                 max_noise_ratio: float = 0.3) -> bool:
    
    
    if np.std(signal) < min_std:
        return False
    
    
    noise_level = np.std(np.diff(signal)) / np.std(signal)
    if noise_level > max_noise_ratio:
        return False
    
    return True
