# =============================================================================
# Dataset Creation and Preprocessing
# =============================================================================
"""
Data loading, preprocessing, and PyTorch Dataset creation for Telemanom.

This module handles:
- Loading raw telemetry data from NASA Telemanom dataset
- MinMax normalization
- Sliding window sequence creation
- PyTorch DataLoader generation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TelemetryDataset(Dataset):
    """
    PyTorch Dataset for satellite telemetry sequences.

    Handles windowed time series data for LSTM-based anomaly detection.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ) -> None:
        """
        Initialize the telemetry dataset.

        Args:
            sequences: Array of shape (n_samples, window_size, n_features)
            labels: Array of shape (n_samples,) with binary labels
            transform: Optional transform to apply to sequences
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

        logger.info(
            f"TelemetryDataset initialized: {len(self)} samples, "
            f"sequence shape: {self.sequences.shape[1:]}"
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (sequence, label) tensors
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]

        if self.transform:
            sequence = self.transform(sequence)

        return sequence, label


def load_telemanom_data(
    data_dir: Union[str, Path],
    channels: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NASA Telemanom dataset.

    The Telemanom dataset contains telemetry data from the Soil Moisture
    Active Passive (SMAP) satellite and the Mars Science Laboratory (MSL) rover.

    Args:
        data_dir: Path to the raw data directory
        channels: List of channel IDs to load (default: all available)

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    data_dir = Path(data_dir)

    # Check if we have the standard Telemanom structure
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    labels_file = data_dir / "labeled_anomalies.csv"

    if train_dir.exists() and test_dir.exists():
        logger.info(f"Loading Telemanom data from {data_dir}")
        return _load_telemanom_standard(train_dir, test_dir, labels_file, channels)

    # Fallback: Generate synthetic data for demo/testing
    logger.warning("Telemanom data not found. Generating synthetic data for demonstration.")
    return _generate_synthetic_data()


def _load_telemanom_standard(
    train_dir: Path,
    test_dir: Path,
    labels_file: Path,
    channels: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from standard Telemanom directory structure.

    Args:
        train_dir: Directory containing training .npy files
        test_dir: Directory containing test .npy files
        labels_file: CSV file with anomaly labels
        channels: Channels to load

    Returns:
        Tuple of arrays (train_data, train_labels, test_data, test_labels)
    """
    import pandas as pd

    # Load anomaly labels
    if labels_file.exists():
        labels_df = pd.read_csv(labels_file)
        anomaly_info = {
            row["chan_id"]: eval(row["anomaly_sequences"])
            for _, row in labels_df.iterrows()
        }
    else:
        logger.warning("Labels file not found. Using all-normal labels.")
        anomaly_info = {}

    all_train_data = []
    all_train_labels = []
    all_test_data = []
    all_test_labels = []

    # Get available channels
    available_channels = [f.stem for f in train_dir.glob("*.npy")]

    if channels is None:
        channels = available_channels
    else:
        channels = [c for c in channels if c in available_channels]

    logger.info(f"Loading {len(channels)} channels: {channels[:5]}...")

    for channel in channels:
        # Load train data
        train_file = train_dir / f"{channel}.npy"
        if train_file.exists():
            train_data = np.load(train_file)
            all_train_data.append(train_data)
            # Training data is assumed to be normal
            all_train_labels.append(np.zeros(len(train_data), dtype=np.int64))

        # Load test data
        test_file = test_dir / f"{channel}.npy"
        if test_file.exists():
            test_data = np.load(test_file)
            all_test_data.append(test_data)

            # Create labels based on anomaly info
            test_labels = np.zeros(len(test_data), dtype=np.int64)
            if channel in anomaly_info:
                for start, end in anomaly_info[channel]:
                    test_labels[start:end] = 1
            all_test_labels.append(test_labels)

    # Concatenate all channels
    train_data = np.concatenate(all_train_data, axis=0)
    train_labels = np.concatenate(all_train_labels, axis=0)
    test_data = np.concatenate(all_test_data, axis=0)
    test_labels = np.concatenate(all_test_labels, axis=0)

    logger.info(f"Loaded data - Train: {train_data.shape}, Test: {test_data.shape}")
    logger.info(f"Anomaly ratio in test set: {test_labels.mean():.4f}")

    return train_data, train_labels, test_data, test_labels


def _generate_synthetic_data(
    n_train_samples: int = 10000,
    n_test_samples: int = 2000,
    n_features: int = 25,
    anomaly_ratio: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic telemetry data for demonstration.

    Creates realistic-looking satellite telemetry data with:
    - Multiple correlated features
    - Periodic patterns (orbital cycles)
    - Noise
    - Injected anomalies (point and collective)

    Args:
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
        n_features: Number of telemetry features
        anomaly_ratio: Proportion of anomalous samples in test set

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    np.random.seed(42)

    def generate_normal_telemetry(n_samples: int, n_features: int) -> np.ndarray:
        """Generate normal telemetry patterns."""
        t = np.linspace(0, 10 * np.pi, n_samples)

        # Base signals with different frequencies (orbital, diurnal, etc.)
        base_signals = np.column_stack([
            np.sin(t * (i + 1) / 10) + np.random.normal(0, 0.1, n_samples)
            for i in range(n_features // 3)
        ])

        # Temperature-like signals
        temp_signals = np.column_stack([
            20 + 5 * np.sin(t / 5) + np.random.normal(0, 0.5, n_samples)
            for _ in range(n_features // 3)
        ])

        # Power/voltage signals
        power_signals = np.column_stack([
            28 + 0.5 * np.sin(t / 3) + np.random.normal(0, 0.1, n_samples)
            for _ in range(n_features - 2 * (n_features // 3))
        ])

        return np.hstack([base_signals, temp_signals, power_signals])

    # Generate training data (all normal)
    train_data = generate_normal_telemetry(n_train_samples, n_features)
    train_labels = np.zeros(n_train_samples, dtype=np.int64)

    # Generate test data with anomalies
    test_data = generate_normal_telemetry(n_test_samples, n_features)
    test_labels = np.zeros(n_test_samples, dtype=np.int64)

    # Inject anomalies
    n_anomalies = int(n_test_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_test_samples, n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(["spike", "drift", "dropout"])

        if anomaly_type == "spike":
            # Sudden spike in values
            test_data[idx] = test_data[idx] * np.random.uniform(3, 5)
        elif anomaly_type == "drift":
            # Gradual drift from normal
            test_data[idx] = test_data[idx] + np.random.uniform(5, 10)
        else:
            # Sensor dropout
            test_data[idx] = np.zeros(n_features)

        test_labels[idx] = 1

    logger.info(
        f"Generated synthetic data - Train: {train_data.shape}, Test: {test_data.shape}"
    )
    logger.info(f"Injected {n_anomalies} anomalies ({anomaly_ratio*100:.1f}%)")

    return train_data, train_labels, test_data, test_labels


def normalize_data(
    train_data: np.ndarray,
    test_data: np.ndarray,
    method: str = "minmax"
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Normalize data using MinMax scaling.

    Fits the scaler on training data only to prevent data leakage.

    Args:
        train_data: Training data array
        test_data: Test data array
        method: Normalization method (currently only "minmax" supported)

    Returns:
        Tuple of (normalized_train, normalized_test, fitted_scaler)
    """
    if method != "minmax":
        raise ValueError(f"Unknown normalization method: {method}")

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit only on training data
    train_normalized = scaler.fit_transform(train_data)
    test_normalized = scaler.transform(test_data)

    logger.info(
        f"Data normalized using {method}. "
        f"Feature range: [{train_normalized.min():.4f}, {train_normalized.max():.4f}]"
    )

    return train_normalized, test_normalized, scaler


def create_sequences(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from time series data.

    Args:
        data: Input data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,)
        window_size: Length of each sequence window
        stride: Step size between consecutive windows

    Returns:
        Tuple of (sequences, sequence_labels)
        - sequences: Array of shape (n_windows, window_size, n_features)
        - sequence_labels: Array of shape (n_windows,)
    """
    n_samples = len(data)
    n_features = data.shape[1] if len(data.shape) > 1 else 1

    # Ensure data is 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    sequences = []
    sequence_labels = []

    for i in range(0, n_samples - window_size + 1, stride):
        seq = data[i:i + window_size]
        sequences.append(seq)

        # Label based on the last timestep in the window
        # Alternative: use majority vote or any anomaly in window
        seq_label = labels[i + window_size - 1]
        sequence_labels.append(seq_label)

    sequences = np.array(sequences, dtype=np.float32)
    sequence_labels = np.array(sequence_labels, dtype=np.int64)

    logger.info(
        f"Created {len(sequences)} sequences of shape {sequences.shape[1:]}"
    )
    logger.info(
        f"Label distribution: Normal={np.sum(sequence_labels==0)}, "
        f"Anomaly={np.sum(sequence_labels==1)}"
    )

    return sequences, sequence_labels


def get_data_loaders(
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    test_sequences: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders with STRATIFIED splitting.

    IMPORTANT: This function MERGES train and test sequences, then performs
    a stratified split to ensure anomalies are present in ALL splits.
    This fixes the "Cold Start" problem where original Telemanom has
    0 anomalies in training data.

    Split ratios: 70% Train, 15% Val, 15% Test

    Args:
        train_sequences: Original training sequences array
        train_labels: Original training labels array
        test_sequences: Original test sequences array
        test_labels: Original test labels array
        batch_size: Batch size for all loaders
        val_ratio: (ignored - using fixed 70/15/15 split)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for reproducible splitting

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    # =========================================================================
    # MERGE all data to fix Cold Start problem
    # =========================================================================
    logger.info("=" * 50)
    logger.info("STRATIFIED SPLIT: Merging train + test sequences")
    logger.info("=" * 50)

    # Concatenate all sequences and labels
    all_sequences = np.concatenate([train_sequences, test_sequences], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    total_samples = len(all_labels)
    total_anomalies = np.sum(all_labels == 1)
    total_normal = np.sum(all_labels == 0)

    logger.info(f"Total merged data: {total_samples} samples")
    logger.info(f"  - Normal: {total_normal}")
    logger.info(f"  - Anomaly: {total_anomalies}")
    logger.info(f"  - Anomaly ratio: {total_anomalies/total_samples*100:.2f}%")

    # =========================================================================
    # STRATIFIED SPLIT: 70% Train, 15% Val, 15% Test
    # =========================================================================
    # First split: 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_sequences,
        all_labels,
        test_size=0.30,
        random_state=seed,
        stratify=all_labels  # CRITICAL: Ensures anomaly distribution
    )

    # Second split: 50% of temp = 15% val, 50% of temp = 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp  # CRITICAL: Ensures anomaly distribution
    )

    # =========================================================================
    # LOG ANOMALY DISTRIBUTION (Critical for debugging)
    # =========================================================================
    train_anomalies = np.sum(y_train == 1)
    val_anomalies = np.sum(y_val == 1)
    test_anomalies = np.sum(y_test == 1)

    logger.info("-" * 50)
    logger.info("STRATIFIED SPLIT RESULT:")
    logger.info(f"  Train: {len(y_train)} samples (Normal={np.sum(y_train==0)}, Anomaly={train_anomalies})")
    logger.info(f"  Val:   {len(y_val)} samples (Normal={np.sum(y_val==0)}, Anomaly={val_anomalies})")
    logger.info(f"  Test:  {len(y_test)} samples (Normal={np.sum(y_test==0)}, Anomaly={test_anomalies})")
    logger.info("-" * 50)

    # Verify anomalies are in training set
    if train_anomalies == 0:
        logger.error("CRITICAL: Still 0 anomalies in training set after stratified split!")
        logger.error("Check if your dataset has enough anomalies for stratification.")
    else:
        logger.info(f"SUCCESS: {train_anomalies} anomalies in training set!")

    # =========================================================================
    # CREATE DATASETS AND DATALOADERS
    # =========================================================================
    train_dataset = TelemetryDataset(X_train, y_train)
    val_dataset = TelemetryDataset(X_val, y_val)
    test_dataset = TelemetryDataset(X_test, y_test)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }

    return loaders


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Uses inverse frequency weighting to handle class imbalance.

    Args:
        labels: Array of class labels

    Returns:
        torch.Tensor: Class weights tensor
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    # Inverse frequency weighting
    weights = total / (len(unique) * counts)

    logger.info(f"Computed class weights: {dict(zip(unique, weights))}")

    return torch.FloatTensor(weights)
