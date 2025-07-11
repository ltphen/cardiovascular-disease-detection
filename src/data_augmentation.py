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
=======
import tensorflow as tf
from scipy import signal
from typing import Tuple, Optional, List
import random

class ECGAugmentor:
    """
    ECG-specific data augmentation class with various signal processing techniques.
    """
    
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        
    def add_noise(self, ecg_signal: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            noise_factor: Noise intensity factor
            
        Returns:
            Noisy ECG signal
        """
        noise = np.random.normal(0, noise_factor * np.std(ecg_signal), ecg_signal.shape)
        return ecg_signal + noise
    
    def time_shift(self, ecg_signal: np.ndarray, max_shift: int = 10) -> np.ndarray:
        """
        Apply random time shifting to ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            max_shift: Maximum number of samples to shift
            
        Returns:
            Time-shifted ECG signal
        """
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return np.pad(ecg_signal, (shift, 0), mode='constant')[:-shift]
        elif shift < 0:
            return np.pad(ecg_signal, (0, -shift), mode='constant')[-shift:]
        else:
            return ecg_signal
    
    def amplitude_scaling(self, ecg_signal: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Apply random amplitude scaling to ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            scale_range: Range of scaling factors
            
        Returns:
            Scaled ECG signal
        """
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return ecg_signal * scale_factor
    
    def baseline_wander(self, ecg_signal: np.ndarray, amplitude: float = 0.1, frequency: float = 0.5) -> np.ndarray:
        """
        Add baseline wander to ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            amplitude: Amplitude of baseline wander
            frequency: Frequency of baseline wander in Hz
            
        Returns:
            ECG signal with baseline wander
        """
        t = np.arange(len(ecg_signal)) / self.sampling_rate
        wander = amplitude * np.sin(2 * np.pi * frequency * t)
        return ecg_signal + wander
    
    def power_line_interference(self, ecg_signal: np.ndarray, amplitude: float = 0.05, frequency: float = 50) -> np.ndarray:
        """
        Add power line interference to ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            amplitude: Amplitude of interference
            frequency: Frequency of interference in Hz (50 or 60)
            
        Returns:
            ECG signal with power line interference
        """
        t = np.arange(len(ecg_signal)) / self.sampling_rate
        interference = amplitude * np.sin(2 * np.pi * frequency * t)
        return ecg_signal + interference
    
    def muscle_artifacts(self, ecg_signal: np.ndarray, amplitude: float = 0.1) -> np.ndarray:
        """
        Add muscle artifacts (EMG) to ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            amplitude: Amplitude of muscle artifacts
            
        Returns:
            ECG signal with muscle artifacts
        """
        # High-frequency noise to simulate muscle artifacts
        muscle_noise = np.random.normal(0, amplitude, ecg_signal.shape)
        # Apply high-pass filter to make it more realistic
        sos = signal.butter(4, 20, btype='high', fs=self.sampling_rate, output='sos')
        muscle_noise = signal.sosfilt(sos, muscle_noise)
        return ecg_signal + muscle_noise
    
    def electrode_motion(self, ecg_signal: np.ndarray, motion_prob: float = 0.1, max_duration: int = 50) -> np.ndarray:
        """
        Simulate electrode motion artifacts.
        
        Args:
            ecg_signal: Input ECG signal
            motion_prob: Probability of motion artifact occurrence
            max_duration: Maximum duration of motion artifact
            
        Returns:
            ECG signal with electrode motion artifacts
        """
        result = ecg_signal.copy()
        
        if np.random.random() < motion_prob:
            # Random location and duration for motion artifact
            start_idx = np.random.randint(0, len(ecg_signal) - max_duration)
            duration = np.random.randint(10, max_duration)
            
            # Create motion artifact (sudden spike or step)
            artifact_type = np.random.choice(['spike', 'step'])
            if artifact_type == 'spike':
                artifact = np.random.normal(0, 2 * np.std(ecg_signal), duration)
            else:  # step
                step_height = np.random.normal(0, np.std(ecg_signal))
                artifact = np.full(duration, step_height)
            
            result[start_idx:start_idx + duration] += artifact
        
        return result
    
    def frequency_shift(self, ecg_signal: np.ndarray, shift_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """
        Apply frequency domain shifting (time warping).
        
        Args:
            ecg_signal: Input ECG signal
            shift_range: Range of frequency shift factors
            
        Returns:
            Frequency-shifted ECG signal
        """
        shift_factor = np.random.uniform(shift_range[0], shift_range[1])
        original_length = len(ecg_signal)
        
        # Resample to simulate frequency shift
        new_length = int(original_length / shift_factor)
        resampled = signal.resample(ecg_signal, new_length)
        
        # Crop or pad to maintain original length
        if len(resampled) > original_length:
            start_idx = (len(resampled) - original_length) // 2
            return resampled[start_idx:start_idx + original_length]
        else:
            pad_length = original_length - len(resampled)
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            return np.pad(resampled, (pad_left, pad_right), mode='edge')
    
    def smooth_signal(self, ecg_signal: np.ndarray, window_length: int = 5) -> np.ndarray:
        """
        Apply smoothing to ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            window_length: Length of smoothing window
            
        Returns:
            Smoothed ECG signal
        """
        if window_length % 2 == 0:
            window_length += 1
        
        return signal.savgol_filter(ecg_signal, window_length, 2)
    
    def apply_augmentation(self, ecg_signal: np.ndarray, augmentation_list: List[str], 
                          augmentation_prob: float = 0.5) -> np.ndarray:
        """
        Apply a combination of augmentations to ECG signal.
        
        Args:
            ecg_signal: Input ECG signal
            augmentation_list: List of augmentation techniques to apply
            augmentation_prob: Probability of applying each augmentation
            
        Returns:
            Augmented ECG signal
        """
        result = ecg_signal.copy()
        
        for aug_type in augmentation_list:
            if np.random.random() < augmentation_prob:
                if aug_type == 'noise':
                    result = self.add_noise(result, noise_factor=np.random.uniform(0.05, 0.15))
                elif aug_type == 'time_shift':
                    result = self.time_shift(result, max_shift=int(0.05 * self.sampling_rate))
                elif aug_type == 'amplitude_scaling':
                    result = self.amplitude_scaling(result, scale_range=(0.8, 1.2))
                elif aug_type == 'baseline_wander':
                    result = self.baseline_wander(result, amplitude=np.random.uniform(0.05, 0.15))
                elif aug_type == 'power_line':
                    result = self.power_line_interference(result, amplitude=np.random.uniform(0.02, 0.08))
                elif aug_type == 'muscle_artifacts':
                    result = self.muscle_artifacts(result, amplitude=np.random.uniform(0.05, 0.15))
                elif aug_type == 'electrode_motion':
                    result = self.electrode_motion(result, motion_prob=0.1)
                elif aug_type == 'frequency_shift':
                    result = self.frequency_shift(result, shift_range=(0.95, 1.05))
                elif aug_type == 'smooth':
                    result = self.smooth_signal(result, window_length=np.random.choice([3, 5, 7]))
        
        return result

def create_augmented_dataset(X: np.ndarray, y: np.ndarray, augmentation_factor: int = 2, 
                           sampling_rate: int = 100, augmentation_list: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an augmented dataset from original ECG data.
    
    Args:
        X: Original ECG data of shape (n_samples, seq_len, features)
        y: Original labels
        augmentation_factor: Number of augmented samples per original sample
        sampling_rate: Sampling rate of ECG signals
        augmentation_list: List of augmentation techniques to use
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    if augmentation_list is None:
        augmentation_list = ['noise', 'time_shift', 'amplitude_scaling', 'baseline_wander', 
                           'power_line', 'muscle_artifacts', 'electrode_motion']
    
    augmentor = ECGAugmentor(sampling_rate)
    
    # Initialize augmented arrays
    n_samples, seq_len, n_features = X.shape
    n_augmented = n_samples * augmentation_factor
    
    X_augmented = np.zeros((n_augmented, seq_len, n_features))
    y_augmented = np.zeros((n_augmented, y.shape[1]))
    
    # Copy original data
    X_augmented[:n_samples] = X
    y_augmented[:n_samples] = y
    
    # Generate augmented samples
    for i in range(n_samples):
        for j in range(1, augmentation_factor):
            idx = i * augmentation_factor + j
            
            # Apply augmentation to each feature (lead)
            for feature in range(n_features):
                X_augmented[idx, :, feature] = augmentor.apply_augmentation(
                    X[i, :, feature], 
                    augmentation_list,
                    augmentation_prob=0.7
                )
            
            # Copy labels
            y_augmented[idx] = y[i]
    
    return X_augmented, y_augmented

def preprocess_ecg_advanced(ecg_signal: np.ndarray, sampling_rate: int = 100, 
                          apply_filters: bool = True) -> np.ndarray:
    """
    Advanced preprocessing for ECG signals.
    
    Args:
        ecg_signal: Input ECG signal
        sampling_rate: Sampling rate of the signal
        apply_filters: Whether to apply filtering

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

