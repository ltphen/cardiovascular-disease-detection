import numpy as np
from typing import Tuple

"""Signal augmentation utilities for ECG time-series.
These methods operate on numpy arrays of shape (n_samples, seq_len, n_channels).
All functions assume the signals are already float32 / float64.
"""

def random_jitter(signal: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Add Gaussian noise (jitter) to a 1D ECG signal."""
    noise = np.random.normal(0, sigma, size=signal.shape)
    return signal + noise


def random_scaling(signal: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Randomly scale the amplitude of the ECG signal."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale


def random_time_shift(signal: np.ndarray, shift_range: int = 100) -> np.ndarray:
    """Circularly shift (roll) the ECG signal in time.

    shift_range is the maximum absolute number of samples to shift.
    """
    shift = np.random.randint(-shift_range, shift_range + 1)
    return np.roll(signal, shift, axis=0)


def mixup_signals(
    signal1: np.ndarray,
    label1: np.ndarray,
    signal2: np.ndarray,
    label2: np.ndarray,
    alpha: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform mixup between two signals and their multi-label targets."""
    lam = np.random.beta(alpha, alpha)
    mixed_signal = lam * signal1 + (1 - lam) * signal2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_signal, mixed_label


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    jitter_sigma: float = 0.03,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    shift_range: int = 100,
    mixup_alpha: float = 0.2,
    mixup_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an augmented dataset with several augmentation techniques.

    Parameters
    ----------
    X : ndarray, shape (n_samples, seq_len, n_channels)
    y : ndarray, shape (n_samples, n_labels)
    jitter_sigma : std-dev for Gaussian jitter
    scale_range : tuple of (min_scale, max_scale)
    shift_range : max absolute time shift in samples
    mixup_alpha : concentration parameter for Beta distribution in mixup
    mixup_ratio : fraction of augmented samples generated via mixup relative to original set size

    Returns
    -------
    X_aug, y_aug : augmented samples (NOT including the originals)
    """
    n_samples = X.shape[0]
    augmented_X = []
    augmented_y = []

    # --- standard stochastic augmentations (jitter, scaling, shift) ---
    for i in range(n_samples):
        sig = X[i].copy()
        lbl = y[i].copy()

        sig = random_jitter(sig, sigma=jitter_sigma)
        sig = random_scaling(sig, scale_range=scale_range)
        sig = random_time_shift(sig, shift_range=shift_range)

        # Optional re-normalize (z-score) to maintain similar scale
        sig_flat = sig[:, 0]
        sig_flat = (sig_flat - np.mean(sig_flat)) / (np.std(sig_flat) + 1e-8)
        sig[:, 0] = sig_flat

        augmented_X.append(sig)
        augmented_y.append(lbl)

    # --- mixup augmentations ---
    n_mixup = int(n_samples * mixup_ratio)
    if n_mixup > 0:
        idx1 = np.random.randint(0, n_samples, size=n_mixup)
        idx2 = np.random.randint(0, n_samples, size=n_mixup)
        for i1, i2 in zip(idx1, idx2):
            mixed_sig, mixed_lbl = mixup_signals(
                X[i1], y[i1], X[i2], y[i2], alpha=mixup_alpha
            )
            augmented_X.append(mixed_sig)
            augmented_y.append(mixed_lbl)

    X_aug = np.stack(augmented_X, axis=0)
    y_aug = np.stack(augmented_y, axis=0)

    return X_aug, y_aug