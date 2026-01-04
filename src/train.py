# =============================================================================
# Training Script
# =============================================================================
"""
Production-ready training pipeline for LSTM anomaly detector.

Features:
- Hydra configuration management
- WandB experiment tracking
- Early stopping and learning rate scheduling
- Best model checkpointing
- Comprehensive logging
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import (
    compute_class_weights,
    create_sequences,
    get_data_loaders,
    load_telemanom_data,
    normalize_data,
)
from src.models.lstm_model import LSTMAnomalyDetector
from src.utils import (
    AverageMeter,
    EarlyStopping,
    get_device,
    save_checkpoint,
    set_seed,
    setup_logging,
    count_parameters,
)

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Computing device
        gradient_clip: Maximum gradient norm

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()

    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Accuracy")

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, (sequences, labels) in enumerate(pbar):
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(sequences)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Compute metrics
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean().item()

        # Update meters
        batch_size = sequences.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy, batch_size)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.4f}"
        })

    return loss_meter.avg, acc_meter.avg


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Computing device

    Returns:
        Dictionary of validation metrics
    """
    model.eval()

    loss_meter = AverageMeter("Loss")
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validating", leave=False):
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            logits, _ = model(sequences)
            loss = criterion(logits, labels)

            # Store predictions
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss_meter.update(loss.item(), sequences.size(0))

    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    metrics = {
        "val_loss": loss_meter.avg,
        "val_accuracy": (all_predictions == all_labels).mean(),
        "val_f1": f1_score(all_labels, all_predictions, average="weighted", zero_division=0),
        "val_precision": precision_score(all_labels, all_predictions, average="weighted", zero_division=0),
        "val_recall": recall_score(all_labels, all_predictions, average="weighted", zero_division=0),
        "val_f1_anomaly": f1_score(all_labels, all_predictions, pos_label=1, zero_division=0),
    }

    return metrics


def create_optimizer(
    model: nn.Module,
    cfg: DictConfig
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model: Neural network model
        cfg: Configuration object

    Returns:
        Configured optimizer
    """
    if cfg.optimizer.name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.training.weight_decay
        )
    elif cfg.optimizer.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.training.weight_decay
        )
    elif cfg.optimizer.name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=0.9,
            weight_decay=cfg.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer
        cfg: Configuration object

    Returns:
        Configured scheduler or None
    """
    if cfg.scheduler.name.lower() == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Maximize F1 score
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
            min_lr=cfg.scheduler.min_lr,
            verbose=True
        )
    elif cfg.scheduler.name.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.scheduler.min_lr
        )
    elif cfg.scheduler.name.lower() == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler.name}")

    return scheduler


def init_wandb(cfg: DictConfig) -> None:
    """
    Initialize Weights & Biases tracking.

    Args:
        cfg: Configuration object
    """
    if cfg.wandb.enabled:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.run_name,
                tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
                config=OmegaConf.to_container(cfg, resolve=True)
            )
            logger.info("WandB initialized successfully")
        except ImportError:
            logger.warning("WandB not installed. Skipping experiment tracking.")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")


def log_wandb(metrics: Dict[str, float], step: int) -> None:
    """
    Log metrics to WandB.

    Args:
        metrics: Dictionary of metrics
        step: Current step/epoch
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main training function.

    Args:
        cfg: Hydra configuration

    Returns:
        Best validation F1 score
    """
    # Setup logging
    setup_logging(
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level
    )

    logger.info("=" * 60)
    logger.info("Tiny-Sat-Anomaly Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seeds for reproducibility
    set_seed(cfg.seed)

    # Get device
    device = get_device(cfg.device)

    # Initialize WandB
    init_wandb(cfg)

    # =========================================================================
    # Data Loading
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Loading and preprocessing data...")
    logger.info("=" * 40)

    # Load raw data
    train_data, train_labels, test_data, test_labels = load_telemanom_data(
        data_dir=cfg.data.raw_path,
        channels=cfg.data.channels if cfg.data.channels else None
    )

    # Normalize data
    train_data_norm, test_data_norm, scaler = normalize_data(
        train_data, test_data, method="minmax"
    )

    # Create sequences
    train_sequences, train_seq_labels = create_sequences(
        train_data_norm,
        train_labels,
        window_size=cfg.training.window_size
    )

    test_sequences, test_seq_labels = create_sequences(
        test_data_norm,
        test_labels,
        window_size=cfg.training.window_size
    )

    # Create data loaders
    loaders = get_data_loaders(
        train_sequences=train_sequences,
        train_labels=train_seq_labels,
        test_sequences=test_sequences,
        test_labels=test_seq_labels,
        batch_size=cfg.training.batch_size,
        val_ratio=cfg.data.val_ratio,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        seed=cfg.seed
    )

    # =========================================================================
    # Model Setup
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Initializing model...")
    logger.info("=" * 40)

    # Update input size based on actual data
    actual_input_size = train_sequences.shape[2]
    if actual_input_size != cfg.model.input_size:
        logger.warning(
            f"Config input_size ({cfg.model.input_size}) differs from data ({actual_input_size}). "
            f"Using data dimension."
        )

    model = LSTMAnomalyDetector(
        input_size=actual_input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional
    ).to(device)

    logger.info(f"\n{model}")
    logger.info(f"Total trainable parameters: {count_parameters(model):,}")

    # =========================================================================
    # Loss, Optimizer, Scheduler
    # =========================================================================
    # Compute class weights for imbalanced data
    if cfg.training.class_weights:
        class_weights = torch.tensor(cfg.training.class_weights, dtype=torch.float32)
    else:
        class_weights = compute_class_weights(train_seq_labels)

    class_weights = class_weights.to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Weighted cross entropy loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = create_optimizer(model, cfg)

    # Scheduler
    scheduler = create_scheduler(optimizer, cfg)

    # Early stopping
    early_stopping = EarlyStopping(
        patience=cfg.training.early_stopping_patience,
        mode=cfg.checkpoint.mode
    )

    # =========================================================================
    # Training Loop
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("Starting training...")
    logger.info("=" * 40)

    best_f1 = 0.0
    best_epoch = 0

    checkpoint_dir = Path(cfg.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.training.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{cfg.training.epochs}")
        logger.info("-" * 30)

        # Training
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip=cfg.training.gradient_clip
        )

        # Validation
        val_metrics = validate(
            model=model,
            dataloader=loaders["val"],
            criterion=criterion,
            device=device
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        logger.info(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val F1: {val_metrics['val_f1']:.4f} | "
            f"Val F1 (Anomaly): {val_metrics['val_f1_anomaly']:.4f} | LR: {current_lr:.2e}"
        )

        # Log to WandB
        log_wandb({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_metrics["val_loss"],
            "val/accuracy": val_metrics["val_accuracy"],
            "val/f1": val_metrics["val_f1"],
            "val/f1_anomaly": val_metrics["val_f1_anomaly"],
            "val/precision": val_metrics["val_precision"],
            "val/recall": val_metrics["val_recall"],
            "learning_rate": current_lr,
        }, step=epoch)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["val_f1"])
            else:
                scheduler.step()

        # Save best model based on F1-Score for ANOMALY class (not weighted F1)
        # This is critical for imbalanced anomaly detection
        current_f1_anomaly = val_metrics["val_f1_anomaly"]
        if current_f1_anomaly > best_f1:
            best_f1 = current_f1_anomaly
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                filepath=checkpoint_dir / "best_model.pth",
                scheduler=scheduler
            )
            logger.info(f"New best model saved! F1 (Anomaly): {best_f1:.4f}")

        # Early stopping check based on F1 (Anomaly class)
        if early_stopping(current_f1_anomaly):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # =========================================================================
    # Final Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best F1 Score (Anomaly Class): {best_f1:.4f} (Epoch {best_epoch})")
    logger.info(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")

    # Close WandB
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    return best_f1


if __name__ == "__main__":
    main()
