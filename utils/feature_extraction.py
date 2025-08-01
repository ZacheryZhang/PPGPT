import numpy as np
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import welch, find_peaks
from scipy.ndimage import uniform_filter1d

def extract_time_features(signal: np.ndarray) -> np.ndarray:
    
    features = []
    
    
    features.extend([
        np.mean(signal),
        np.std(signal), 
        np.var(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal),
        skew(signal),
        kurtosis(signal)
    ])
    
    
    peaks, _ = find_peaks(signal, height=np.mean(signal))
    features.extend([
        len(peaks),  
        np.mean(np.diff(peaks)) if len(peaks) > 1 else 0,  
    ])
    
    return np.array(features)

def extract_frequency_features(signal: np.ndarray, fs: int = 64) -> np.ndarray:
    
    
    freqs, psd = welch(signal, fs, nperseg=min(len(signal)//2, 256))
    
    features = []
    
    
    hr_band = (freqs >= 0.7) & (freqs <= 4.0)  
    resp_band = (freqs >= 0.15) & (freqs <= 0.5)  
    
    features.extend([
        np.sum(psd[hr_band]),   
        np.sum(psd[resp_band]), 
        np.sum(psd),            
        entropy(psd),           
        freqs[np.argmax(psd)]   
    ])
    
    return np.array(features)

def extract_morphological_features(signal: np.ndarray) -> np.ndarray:
    
    features = []
    
    
    smoothed = uniform_filter1d(signal, size=3)
    
    
    diff1 = np.diff(smoothed)
    features.extend([
        np.mean(np.abs(diff1)),
        np.std(diff1)
    ])
    
    
    diff2 = np.diff(diff1)
    features.extend([
        np.mean(np.abs(diff2)),
        np.std(diff2)
    ])
    
    return np.array(features)
