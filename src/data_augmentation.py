import numpy as np
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
        
    Returns:
        Preprocessed ECG signal
    """
    result = ecg_signal.copy()
    
    if apply_filters:
        # High-pass filter to remove baseline wander
        sos_hp = signal.butter(4, 0.5, btype='high', fs=sampling_rate, output='sos')
        result = signal.sosfilt(sos_hp, result)
        
        # Low-pass filter to remove high-frequency noise
        sos_lp = signal.butter(4, 40, btype='low', fs=sampling_rate, output='sos')
        result = signal.sosfilt(sos_lp, result)
        
        # Notch filter to remove power line interference
        for freq in [50, 60]:  # Common power line frequencies
            sos_notch = signal.iirnotch(freq, 30, fs=sampling_rate)
            result = signal.sosfilt(sos_notch, result)
    
    # Z-score normalization
    result = (result - np.mean(result)) / (np.std(result) + 1e-8)
    
    return result

def augment_batch_tf(batch_x: tf.Tensor, batch_y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow-based batch augmentation for training.
    
    Args:
        batch_x: Batch of ECG signals
        batch_y: Batch of labels
        
    Returns:
        Augmented batch
    """
    # Add random noise
    noise = tf.random.normal(tf.shape(batch_x), stddev=0.1)
    batch_x_aug = batch_x + noise
    
    # Random amplitude scaling
    scale = tf.random.uniform([tf.shape(batch_x)[0], 1, 1], 0.8, 1.2)
    batch_x_aug = batch_x_aug * scale
    
    # Random time shift (simple version)
    # This is a simplified version - for full implementation, use tf.image.random_crop and pad
    
    return batch_x_aug, batch_y