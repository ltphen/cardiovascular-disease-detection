import numpy as np
from scipy.stats import skew, kurtosis
from typing import List, Optional

def extract_time_features(window: np.ndarray, fs: int = 500) -> np.ndarray:
    """
    Extract time-domain features from an ECG window.
    
    Args:
        window: 1D numpy array of ECG signal (already z-scored)
        fs: Sampling frequency in Hz
        
    Returns:
        1D numpy array of features
    """
    feat = []
    
    # Basic statistics
    feat.append(np.mean(window))        # should be ~0 if z-scored
    feat.append(np.std(window))
    feat.append(np.max(window))         # peak amplitude
    feat.append(np.min(window))         # trough amplitude
    feat.append(skew(window))           # skewness
    feat.append(kurtosis(window))       # kurtosis
    
    # Peak-to-peak amplitude
    feat.append(np.max(window) - np.min(window))
    
    # QRS width estimate
    r_idx = np.argmax(window)
    qrs_region = window[max(0, r_idx - 50):min(len(window), r_idx + 50)]
    half_peak = 0.5 * window[r_idx]
    qrs_width = np.sum(qrs_region > half_peak) / fs  # in seconds
    feat.append(qrs_width)
    
    # Area under absolute curve
    feat.append(np.sum(np.abs(window)) / fs)  # approximate area, scaled by time
    
    # Additional time-domain features
    # 1. Mean absolute deviation
    feat.append(np.mean(np.abs(window - np.mean(window))))
    
    # 2. Root mean square
    feat.append(np.sqrt(np.mean(window ** 2)))
    
    # 3. Zero crossing rate
    feat.append(np.sum(np.diff(np.signbit(window))) / len(window))
    
    # 4. Signal energy
    feat.append(np.sum(window ** 2) / len(window))
    
    return np.array(feat)

def extract_frequency_features(window: np.ndarray, fs: int = 500) -> np.ndarray:
    """
    Extract frequency-domain features from an ECG window.
    
    Args:
        window: 1D numpy array of ECG signal
        fs: Sampling frequency in Hz
        
    Returns:
        1D numpy array of features
    """
    # Compute FFT
    fft_vals = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), 1/fs)
    
    # Define frequency bands (in Hz)
    bands = {
        'delta': (0, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    feat = []
    
    # Energy in each frequency band
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_energy = np.sum(fft_vals[mask] ** 2)
        feat.append(band_energy)
    
    # Spectral entropy
    psd = fft_vals ** 2
    psd = psd / np.sum(psd)  # normalize
    spectral_entropy = -np.sum(psd * np.log2(psd + 1e-10))
    feat.append(spectral_entropy)
    
    # Dominant frequency
    dominant_freq = freqs[np.argmax(fft_vals)]
    feat.append(dominant_freq)
    
    # Spectral centroid
    centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    feat.append(centroid)
    
    return np.array(feat)

def extract_features(
    windows: np.ndarray,
    fs: int = 500,
    include_freq: bool = True
) -> np.ndarray:
    """
    Extract features from a batch of ECG windows.
    
    Args:
        windows: Array of shape (n_windows, window_size, 1)
        fs: Sampling frequency in Hz
        include_freq: Whether to include frequency-domain features
        
    Returns:
        Array of shape (n_windows, n_features)
    """
    n_windows = windows.shape[0]
    features = []
    
    for i in range(n_windows):
        window = windows[i, :, 0]  # shape: (window_size,)
        
        # Extract time-domain features
        time_feat = extract_time_features(window, fs)
        
        if include_freq:
            # Extract frequency-domain features
            freq_feat = extract_frequency_features(window, fs)
            # Combine features
            feat = np.concatenate([time_feat, freq_feat])
        else:
            feat = time_feat
            
        features.append(feat)
        
    return np.stack(features, axis=0) 