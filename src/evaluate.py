# =============================================================================
# Evaluation Script
# =============================================================================
"""
Model evaluation and anomaly visualization.

Features:
- Comprehensive classification metrics (F1, Precision, Recall, etc.)
- Confusion matrix visualization
- Anomaly detection timeline plot
- Threshold optimization for anomaly detection
- Detailed per-class analysis
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import (
    create_sequences,
    get_data_loaders,
    load_telemanom_data,
    normalize_data,
    TelemetryDataset,
)
from src.models.lstm_model import LSTMAnomalyDetector
from src.utils import get_device, load_checkpoint, set_seed, setup_logging

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model and collect predictions.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Computing device

    Returns:
        Tuple of (labels, predictions, probabilities)
    """
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Evaluating"):
            sequences = sequences.to(device, non_blocking=True)

            # Forward pass
            logits, _ = model(sequences)

            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Anomaly probability

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        labels: Ground truth labels
        predictions: Model predictions
        probabilities: Anomaly probabilities

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_anomaly": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "precision_weighted": precision_score(labels, predictions, average="weighted", zero_division=0),
        "precision_anomaly": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "recall_weighted": recall_score(labels, predictions, average="weighted", zero_division=0),
        "recall_anomaly": recall_score(labels, predictions, pos_label=1, zero_division=0),
    }

    # ROC AUC (only if both classes present)
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def print_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray
) -> None:
    """
    Print detailed classification report.

    Args:
        labels: Ground truth labels
        predictions: Model predictions
    """
    logger.info("\n" + "=" * 50)
    logger.info("Classification Report")
    logger.info("=" * 50)

    report = classification_report(
        labels,
        predictions,
        target_names=["Normal", "Anomaly"],
        digits=4,
        zero_division=0
    )
    logger.info(f"\n{report}")


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
    normalize: bool = True
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        labels: Ground truth labels
        predictions: Model predictions
        output_dir: Directory to save the plot
        normalize: Whether to normalize the matrix
    """
    cm = confusion_matrix(labels, predictions)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".3f" if normalize else "d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()

    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved to: {output_path}")


def plot_anomaly_timeline(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
    window_size: int = 1000
) -> None:
    """
    Plot anomaly detection timeline.

    Shows the anomaly probability over time with ground truth markers.

    Args:
        labels: Ground truth labels
        predictions: Model predictions
        probabilities: Anomaly probabilities
        output_dir: Directory to save the plot
        window_size: Number of samples to display
    """
    # Limit to window_size for readability
    n_samples = min(len(labels), window_size)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    timesteps = np.arange(n_samples)

    # Plot 1: Anomaly probability
    ax1 = axes[0]
    ax1.plot(timesteps, probabilities[:n_samples], color="blue", alpha=0.7, linewidth=0.8)
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold (0.5)")
    ax1.fill_between(timesteps, 0, probabilities[:n_samples], alpha=0.3)
    ax1.set_ylabel("Anomaly\nProbability", fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right")
    ax1.set_title("Anomaly Detection Timeline", fontsize=14, fontweight="bold")

    # Plot 2: Ground truth
    ax2 = axes[1]
    ax2.fill_between(
        timesteps,
        0,
        labels[:n_samples],
        color="green",
        alpha=0.6,
        step="mid",
        label="Ground Truth"
    )
    ax2.set_ylabel("Ground Truth\n(Anomaly=1)", fontsize=10)
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc="upper right")

    # Plot 3: Predictions vs Ground Truth (errors highlighted)
    ax3 = axes[2]

    # True positives
    tp_mask = (labels[:n_samples] == 1) & (predictions[:n_samples] == 1)
    ax3.fill_between(
        timesteps,
        0,
        tp_mask.astype(int),
        color="green",
        alpha=0.6,
        step="mid",
        label="True Positive"
    )

    # False positives
    fp_mask = (labels[:n_samples] == 0) & (predictions[:n_samples] == 1)
    ax3.fill_between(
        timesteps,
        0,
        fp_mask.astype(int),
        color="red",
        alpha=0.6,
        step="mid",
        label="False Positive"
    )

    # False negatives
    fn_mask = (labels[:n_samples] == 1) & (predictions[:n_samples] == 0)
    ax3.fill_between(
        timesteps,
        0,
        -fn_mask.astype(int),
        color="orange",
        alpha=0.6,
        step="mid",
        label="False Negative"
    )

    ax3.set_ylabel("Detection\nResult", fontsize=10)
    ax3.set_xlabel("Time Step", fontsize=12)
    ax3.set_ylim(-1.1, 1.1)
    ax3.legend(loc="upper right", ncol=3)

    plt.tight_layout()

    output_path = output_dir / "anomaly_timeline.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Anomaly timeline saved to: {output_path}")


def plot_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path
) -> None:
    """
    Plot ROC curve.

    Args:
        labels: Ground truth labels
        probabilities: Anomaly probabilities
        output_dir: Directory to save the plot
    """
    if len(np.unique(labels)) < 2:
        logger.warning("Cannot plot ROC curve: only one class present in labels")
        return

    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        fpr, tpr,
        color="blue",
        lw=2,
        label=f"ROC Curve (AUC = {roc_auc:.4f})"
    )
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "roc_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"ROC curve saved to: {output_path}")


def plot_precision_recall_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path
) -> None:
    """
    Plot Precision-Recall curve.

    Args:
        labels: Ground truth labels
        probabilities: Anomaly probabilities
        output_dir: Directory to save the plot
    """
    if len(np.unique(labels)) < 2:
        logger.warning("Cannot plot PR curve: only one class present in labels")
        return

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, color="blue", lw=2)
    ax.fill_between(recall, precision, alpha=0.2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "precision_recall_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Precision-Recall curve saved to: {output_path}")


def find_optimal_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray
) -> Tuple[float, float]:
    """
    Find optimal threshold for anomaly detection based on F1 score.

    Args:
        labels: Ground truth labels
        probabilities: Anomaly probabilities

    Returns:
        Tuple of (optimal_threshold, best_f1_score)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logger.info(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

    return best_threshold, best_f1


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation function.

    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    setup_logging(level=cfg.logging.level)

    logger.info("=" * 60)
    logger.info("Tiny-Sat-Anomaly Evaluation Pipeline")
    logger.info("=" * 60)

    # Set seeds
    set_seed(cfg.seed)

    # Get device
    device = get_device(cfg.device)

    # Create output directory for plots
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Load Data
    # =========================================================================
    logger.info("\nLoading test data...")

    train_data, train_labels, test_data, test_labels = load_telemanom_data(
        data_dir=cfg.data.raw_path,
        channels=cfg.data.channels if cfg.data.channels else None
    )

    # Normalize using training statistics
    train_data_norm, test_data_norm, scaler = normalize_data(
        train_data, test_data, method="minmax"
    )

    # Create test sequences
    test_sequences, test_seq_labels = create_sequences(
        test_data_norm,
        test_labels,
        window_size=cfg.training.window_size
    )

    # Create test dataset and loader
    test_dataset = TelemetryDataset(test_sequences, test_seq_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # =========================================================================
    # Load Model
    # =========================================================================
    logger.info("\nLoading trained model...")

    # Determine input size from data
    actual_input_size = test_sequences.shape[2]

    model = LSTMAnomalyDetector(
        input_size=actual_input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional
    ).to(device)

    # Load checkpoint
    checkpoint_path = Path(cfg.checkpoint.save_dir) / "best_model.pth"

    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, model, device=device)
        logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    else:
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Please train the model first using: python src/train.py")
        return

    # =========================================================================
    # Evaluation
    # =========================================================================
    logger.info("\nRunning evaluation...")

    labels, predictions, probabilities = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device
    )

    # Compute metrics
    metrics = compute_metrics(labels, predictions, probabilities)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)

    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")

    # Detailed classification report
    print_classification_report(labels, predictions)

    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(labels, probabilities)

    # Predictions with optimal threshold
    optimal_predictions = (probabilities >= optimal_threshold).astype(int)
    optimal_metrics = compute_metrics(labels, optimal_predictions, probabilities)

    logger.info("\n" + "=" * 50)
    logger.info(f"Metrics with Optimal Threshold ({optimal_threshold:.2f})")
    logger.info("=" * 50)

    for name, value in optimal_metrics.items():
        logger.info(f"{name}: {value:.4f}")

    # =========================================================================
    # Generate Plots
    # =========================================================================
    logger.info("\nGenerating visualizations...")

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # Generate all plots
    plot_confusion_matrix(labels, predictions, output_dir)
    plot_anomaly_timeline(labels, predictions, probabilities, output_dir)
    plot_roc_curve(labels, probabilities, output_dir)
    plot_precision_recall_curve(labels, probabilities, output_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"\nKey Metrics:")
    logger.info(f"  - F1 Score (Anomaly): {metrics['f1_anomaly']:.4f}")
    logger.info(f"  - Precision (Anomaly): {metrics['precision_anomaly']:.4f}")
    logger.info(f"  - Recall (Anomaly): {metrics['recall_anomaly']:.4f}")
    logger.info(f"  - ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"\nOptimal Threshold: {optimal_threshold:.2f} (F1: {optimal_f1:.4f})")
    logger.info(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
