"""Signal filtering/normalization and label encoding for the ECG pipeline."""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import MultiLabelBinarizer

import config


def filter_signal(signal: np.ndarray, sampling_rate: int = config.SAMPLING_RATE,
                   low_cut: float = 0.5, high_cut: float = 40.0) -> np.ndarray:
    """Bandpass filter (0.5-40Hz) to remove baseline wander and high-frequency noise."""
    nyquist = 0.5 * sampling_rate
    sos = butter(N=4, Wn=[low_cut / nyquist, high_cut / nyquist],
                 btype="bandpass", output="sos")

    filtered = np.zeros_like(signal)
    for lead in range(signal.shape[1]):
        filtered[:, lead] = sosfiltfilt(sos, signal[:, lead])  # zero-phase, avoids time shift
    return filtered


def filter_all_signals(signals: np.ndarray, sampling_rate: int = config.SAMPLING_RATE) -> np.ndarray:
    return np.array([filter_signal(s, sampling_rate) for s in signals])


def normalize_signals(signals: np.ndarray) -> np.ndarray:
    """Per-lead z-score normalization across the batch."""
    mean = signals.mean(axis=(0, 1), keepdims=True)
    std = signals.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    return (signals - mean) / std


def encode_labels(superclass_lists, classes=config.DIAGNOSTIC_CLASSES):
    """Multi-hot encodes diagnostic superclass lists, e.g. ['MI','STTC'] -> [0,1,1,0,0]."""
    mlb = MultiLabelBinarizer(classes=classes)
    encoded = mlb.fit_transform(superclass_lists)
    return encoded, mlb


def split_by_fold(metadata, signals, labels):
    """Uses PTB-XL's patient-stratified fold column rather than a random split."""
    train_mask = metadata.strat_fold < config.VALIDATION_FOLD
    val_mask = metadata.strat_fold == config.VALIDATION_FOLD
    test_mask = metadata.strat_fold == config.TEST_FOLD

    return {
        "X_train": signals[train_mask.values], "y_train": labels[train_mask.values],
        "X_val": signals[val_mask.values], "y_val": labels[val_mask.values],
        "X_test": signals[test_mask.values], "y_test": labels[test_mask.values],
    }
