# =============================================================================
# Utility Functions
# =============================================================================
"""
Core utility functions for reproducibility, logging, and device management.
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Ensures deterministic behavior across:
    - Python's random module
    - NumPy's random generator
    - PyTorch CPU and CUDA operations

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic algorithms (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    logging.info(f"All random seeds set to {seed}")


def get_device(device_config: str = "auto") -> torch.device:
    """
    Get the appropriate computing device.

    Args:
        device_config: Device specification ("auto", "cuda", "cpu")

    Returns:
        torch.device: Selected computing device
    """
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
    else:
        logging.info("Using CPU for computation")

    return device


def setup_logging(
    log_dir: Optional[str] = None,
    level: str = "INFO",
    log_filename: str = "training.log"
) -> logging.Logger:
    """
    Configure the logging system for the project.

    Sets up both console and file logging with proper formatting.

    Args:
        log_dir: Directory for log files (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_filename: Name of the log file

    Returns:
        logging.Logger: Configured root logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir specified)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_filename)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_path / log_filename}")

    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    filepath: Union[str, Path],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        filepath: Path to save the checkpoint
        scheduler: Learning rate scheduler (optional)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> dict:
    """
    Load a training checkpoint.

    Args:
        filepath: Path to the checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load the checkpoint to

    Returns:
        dict: Checkpoint data including epoch and metrics
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logging.info(f"Checkpoint loaded: {filepath} (epoch {checkpoint['epoch']})")

    return checkpoint


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.

    Monitors a metric and stops training if no improvement
    is observed for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max"
    ) -> None:
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "max" for metrics to maximize, "min" for minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            bool: True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return True

        return False


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training epochs.
    """

    def __init__(self, name: str = "Metric") -> None:
        """
        Initialize meter.

        Args:
            name: Name of the metric being tracked
        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update statistics with new value.

        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
