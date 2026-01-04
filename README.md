# Tiny-Sat-Anomaly

Production-grade satellite telemetry anomaly detection using LSTM neural networks.

## Overview

This project implements an LSTM-based anomaly detection system for satellite telemetry data, inspired by NASA's Telemanom dataset. The architecture follows MLOps best practices with modular design, configuration management, and comprehensive logging.

### Key Features

- **LSTM Architecture**: Stacked LSTM layers with dropout regularization
- **Hydra Configuration**: All hyperparameters managed via YAML configs
- **WandB Integration**: Experiment tracking and visualization
- **Reproducibility**: Full seed control for deterministic training
- **Production-Ready**: Type hints, logging, checkpointing, and error handling

## Project Structure

```
tiny-sat-anomaly/
├── configs/
│   └── config.yaml          # Hydra configuration
├── data/
│   ├── raw/                  # Raw Telemanom data
│   └── processed/            # Preprocessed sequences
├── outputs/
│   ├── checkpoints/          # Model checkpoints
│   ├── logs/                 # Training logs
│   └── evaluation/           # Evaluation plots
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py   # Data loading & preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm_model.py     # LSTM architecture
│   ├── __init__.py
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation & visualization
│   └── utils.py              # Utility functions
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd tiny-sat-anomaly
```

### 2. Create virtual environment

```bash
# Using conda (recommended)
conda create -n tiny-sat python=3.10
conda activate tiny-sat

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Download Telemanom dataset

```bash
# Download NASA Telemanom dataset
git clone https://github.com/khundman/telemanom.git temp_telemanom
cp -r temp_telemanom/data/* data/raw/
rm -rf temp_telemanom
```

> **Note**: If the dataset is not available, the system will automatically generate synthetic telemetry data for demonstration.

## Usage

### Training

Train the LSTM anomaly detector:

```bash
# Default configuration
python src/train.py

# Override hyperparameters via CLI
python src/train.py training.learning_rate=0.0005 training.epochs=50

# Use different config
python src/train.py --config-name=config_custom
```

### Evaluation

Evaluate the trained model:

```bash
python src/evaluate.py
```

This generates:
- Classification metrics (F1, Precision, Recall, AUC)
- Confusion matrix plot
- Anomaly timeline visualization
- ROC and Precision-Recall curves

### Configuration Override Examples

```bash
# Change model architecture
python src/train.py model.hidden_size=256 model.num_layers=3

# Adjust training parameters
python src/train.py training.batch_size=128 training.window_size=100

# Disable WandB
python src/train.py wandb.enabled=false

# Use CPU only
python src/train.py device=cpu
```

## Configuration Reference

Key parameters in `configs/config.yaml`:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `model` | `hidden_size` | 128 | LSTM hidden dimension |
| `model` | `num_layers` | 2 | Number of LSTM layers |
| `model` | `dropout` | 0.2 | Dropout rate |
| `training` | `window_size` | 50 | Input sequence length |
| `training` | `batch_size` | 64 | Training batch size |
| `training` | `learning_rate` | 0.001 | Adam learning rate |
| `training` | `epochs` | 100 | Maximum training epochs |
| `training` | `class_weights` | [1.0, 10.0] | Weighted CE loss |

## Model Architecture

```
LSTMAnomalyDetector(
  input_projection: Linear(input_size → hidden_size)
  input_norm: LayerNorm(hidden_size)
  lstm: LSTM(hidden_size, num_layers, dropout=0.2)
  dropout: Dropout(0.2)
  classifier: Sequential(
    Linear(hidden_size → hidden_size/2)
    ReLU()
    Dropout(0.2)
    Linear(hidden_size/2 → 2)
  )
)
```

## Experiment Tracking with WandB

1. Login to WandB:
```bash
wandb login
```

2. Configure in `config.yaml`:
```yaml
wandb:
  enabled: true
  project: "tiny-sat-anomaly"
  entity: "your-username"
```

3. Run training - metrics are automatically logged:
- Training/validation loss curves
- F1, Precision, Recall per epoch
- Learning rate scheduling
- Model hyperparameters

## Output Files

After training:
- `outputs/checkpoints/best_model.pth` - Best model weights
- `outputs/logs/training.log` - Training logs

After evaluation:
- `outputs/evaluation/confusion_matrix.png`
- `outputs/evaluation/anomaly_timeline.png`
- `outputs/evaluation/roc_curve.png`
- `outputs/evaluation/precision_recall_curve.png`

## Reproducibility

All random seeds are fixed for reproducibility:

```python
# In src/utils.py
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

## Performance Metrics

On synthetic demo data (actual results may vary):

| Metric | Value |
|--------|-------|
| F1 Score (Anomaly) | ~0.85 |
| Precision | ~0.82 |
| Recall | ~0.88 |
| ROC AUC | ~0.94 |

## Extending the Project

### Adding New Models

1. Create model in `src/models/new_model.py`
2. Inherit from `nn.Module`
3. Implement `forward()` returning `(logits, hidden)`

### Custom Datasets

1. Implement loader in `src/data/make_dataset.py`
2. Return format: `(train_data, train_labels, test_data, test_labels)`
3. Update config paths

## License

MIT License

## References

- [NASA Telemanom](https://github.com/khundman/telemanom)
- [Detecting Spacecraft Anomalies Using LSTMs](https://arxiv.org/abs/1802.04431)
- [Hydra Documentation](https://hydra.cc/)
- [Weights & Biases](https://wandb.ai/)
