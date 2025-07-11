import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import random
from typing import Tuple, Optional

def add_gaussian_noise(ecg_signal: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
    """
    Add Gaussian noise to ECG signal.
    
    Args:
        ecg_signal: Input ECG signal
        noise_factor: Standard deviation of noise as fraction of signal amplitude
        
    Returns:
        Noisy ECG signal
    """
    noise = np.random.normal(0, noise_factor * np.std(ecg_signal), ecg_signal.shape)
    return ecg_signal + noise

def time_warping(ecg_signal: np.ndarray, warp_factor: float = 0.1) -> np.ndarray:
    """
    Apply time warping to ECG signal.
    
    Args:
        ecg_signal: Input ECG signal
        warp_factor: Maximum warping factor
        
    Returns:
        Time-warped ECG signal
    """
    n_samples = len(ecg_signal)
    time_points = np.linspace(0, 1, n_samples)
    
    # Create warping function
    warp_points = np.linspace(0, 1, 5)
    warp_values = np.random.uniform(-warp_factor, warp_factor, 5)
    warp_values[0] = warp_values[-1] = 0  # Keep endpoints fixed
    
    # Interpolate warping function
    warp_interp = interp1d(warp_points, warp_values, kind='cubic')
    warped_time = time_points + warp_interp(time_points)
    
    # Apply warping
    warped_signal = np.interp(time_points, warped_time, ecg_signal)
    
    return warped_signal

def amplitude_scaling(ecg_signal: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    Scale amplitude of ECG signal.
    
    Args:
        ecg_signal: Input ECG signal
        scale_range: Range for scaling factor
        
    Returns:
        Amplitude-scaled ECG signal
    """
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return ecg_signal * scale_factor

def baseline_wander(ecg_signal: np.ndarray, wander_factor: float = 0.1) -> np.ndarray:
    """
    Add baseline wander to ECG signal.
    
    Args:
        ecg_signal: Input ECG signal
        wander_factor: Maximum baseline wander amplitude
        
    Returns:
        ECG signal with baseline wander
    """
    n_samples = len(ecg_signal)
    
    # Generate low-frequency baseline wander
    freq = np.random.uniform(0.1, 0.5)  # Low frequency
    t = np.linspace(0, n_samples/100, n_samples)  # Time vector
    baseline = wander_factor * np.sin(2 * np.pi * freq * t)
    
    return ecg_signal + baseline

def random_cropping(ecg_signal: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    """
    Randomly crop ECG signal and resize to original length.
    
    Args:
        ecg_signal: Input ECG signal
        crop_ratio: Ratio of signal to keep
        
    Returns:
        Cropped and resized ECG signal
    """
    n_samples = len(ecg_signal)
    crop_length = int(n_samples * crop_ratio)
    
    # Random start point
    start = np.random.randint(0, n_samples - crop_length)
    cropped = ecg_signal[start:start + crop_length]
    
    # Resize to original length using interpolation
    original_indices = np.linspace(0, crop_length - 1, n_samples)
    resized = np.interp(original_indices, np.arange(crop_length), cropped)
    
    return resized

def frequency_domain_augmentation(ecg_signal: np.ndarray, freq_shift: float = 0.1) -> np.ndarray:
    """
    Apply frequency domain augmentation by shifting frequency components.
    
    Args:
        ecg_signal: Input ECG signal
        freq_shift: Maximum frequency shift factor
        
    Returns:
        Frequency-augmented ECG signal
    """
    # FFT
    fft_signal = np.fft.fft(ecg_signal)
    
    # Random frequency shift
    shift = np.random.uniform(-freq_shift, freq_shift)
    n_samples = len(ecg_signal)
    
    # Apply shift in frequency domain
    freq_axis = np.fft.fftfreq(n_samples)
    shifted_fft = fft_signal * np.exp(2j * np.pi * shift * freq_axis)
    
    # Inverse FFT
    augmented_signal = np.real(np.fft.ifft(shifted_fft))
    
    return augmented_signal

def mixup_augmentation(ecg_signal1: np.ndarray, ecg_signal2: np.ndarray, 
                       alpha: float = 0.2) -> Tuple[np.ndarray, float]:
    """
    Apply mixup augmentation between two ECG signals.
    
    Args:
        ecg_signal1: First ECG signal
        ecg_signal2: Second ECG signal
        alpha: Mixup parameter
        
    Returns:
        Tuple of (mixed signal, mixup weight)
    """
    lam = np.random.beta(alpha, alpha)
    mixed_signal = lam * ecg_signal1 + (1 - lam) * ecg_signal2
    
    return mixed_signal, lam

def augment_ecg_batch(ecg_signals: np.ndarray, 
                      augmentation_prob: float = 0.5,
                      noise_factor: float = 0.05,
                      warp_factor: float = 0.1,
                      scale_range: Tuple[float, float] = (0.8, 1.2),
                      wander_factor: float = 0.1,
                      crop_ratio: float = 0.9,
                      freq_shift: float = 0.1) -> np.ndarray:
    """
    Apply multiple augmentation techniques to a batch of ECG signals.
    
    Args:
        ecg_signals: Batch of ECG signals (n_samples, signal_length, channels)
        augmentation_prob: Probability of applying each augmentation
        noise_factor: Gaussian noise factor
        warp_factor: Time warping factor
        scale_range: Amplitude scaling range
        wander_factor: Baseline wander factor
        crop_ratio: Cropping ratio
        freq_shift: Frequency shift factor
        
    Returns:
        Augmented ECG signals
    """
    augmented_signals = ecg_signals.copy()
    
    for i in range(len(ecg_signals)):
        signal = ecg_signals[i, :, 0]  # Assuming single channel
        
        # Apply augmentations with probability
        if np.random.random() < augmentation_prob:
            signal = add_gaussian_noise(signal, noise_factor)
        
        if np.random.random() < augmentation_prob:
            signal = time_warping(signal, warp_factor)
        
        if np.random.random() < augmentation_prob:
            signal = amplitude_scaling(signal, scale_range)
        
        if np.random.random() < augmentation_prob:
            signal = baseline_wander(signal, wander_factor)
        
        if np.random.random() < augmentation_prob:
            signal = random_cropping(signal, crop_ratio)
        
        if np.random.random() < augmentation_prob:
            signal = frequency_domain_augmentation(signal, freq_shift)
        
        augmented_signals[i, :, 0] = signal
    
    return augmented_signals

def create_augmented_dataset(X_train: np.ndarray, y_train: np.ndarray, 
                           augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an augmented dataset by applying various augmentation techniques.
    
    Args:
        X_train: Training ECG signals
        y_train: Training labels
        augmentation_factor: Number of augmented samples per original sample
        
    Returns:
        Tuple of (augmented signals, augmented labels)
    """
    n_samples = len(X_train)
    augmented_X = []
    augmented_y = []
    
    for i in range(n_samples):
        # Add original sample
        augmented_X.append(X_train[i])
        augmented_y.append(y_train[i])
        
        # Add augmented samples
        for _ in range(augmentation_factor):
            augmented_signal = augment_ecg_batch(
                X_train[i:i+1], 
                augmentation_prob=0.7
            )[0]
            
            augmented_X.append(augmented_signal)
            augmented_y.append(y_train[i])
    
    return np.array(augmented_X), np.array(augmented_y)

def apply_signal_preprocessing(ecg_signal: np.ndarray, 
                             fs: int = 100,
                             highpass_freq: float = 0.5,
                             lowpass_freq: float = 40.0) -> np.ndarray:
    """
    Apply signal preprocessing including filtering and normalization.
    
    Args:
        ecg_signal: Input ECG signal
        fs: Sampling frequency
        highpass_freq: High-pass filter frequency
        lowpass_freq: Low-pass filter frequency
        
    Returns:
        Preprocessed ECG signal
    """
    # High-pass filter to remove baseline wander
    b_high, a_high = signal.butter(4, highpass_freq / (fs/2), btype='high')
    filtered_signal = signal.filtfilt(b_high, a_high, ecg_signal)
    
    # Low-pass filter to remove high-frequency noise
    b_low, a_low = signal.butter(4, lowpass_freq / (fs/2), btype='low')
    filtered_signal = signal.filtfilt(b_low, a_low, filtered_signal)
    
    # Normalize signal
    normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    
    return normalized_signal

def preprocess_batch(X_batch: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    Apply preprocessing to a batch of ECG signals.
    
    Args:
        X_batch: Batch of ECG signals
        fs: Sampling frequency
        
    Returns:
        Preprocessed ECG signals
    """
    preprocessed_batch = np.zeros_like(X_batch)
    
    for i in range(len(X_batch)):
        signal = X_batch[i, :, 0]
        preprocessed_signal = apply_signal_preprocessing(signal, fs)
        preprocessed_batch[i, :, 0] = preprocessed_signal
    
    return preprocessed_batch