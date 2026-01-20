# Tiny-Sat-Anomaly

Production-grade satellite telemetry anomaly detection using LSTM neural networks with **real-time web visualization**.

## Overview

This project implements an LSTM-based anomaly detection system for satellite telemetry data, inspired by NASA's Telemanom dataset. The architecture follows MLOps best practices with modular design, configuration management, and comprehensive logging.

### Key Features

- **LSTM Architecture**: Stacked LSTM layers with dropout regularization
- **Hydra Configuration**: All hyperparameters managed via YAML configs
- **WandB Integration**: Experiment tracking and visualization
- **Reproducibility**: Full seed control for deterministic training
- **Production-Ready**: Type hints, logging, checkpointing, and error handling
- **Web Dashboard**: Next.js app with client-side ONNX inference (WebAssembly)
- **Streamlit Dashboard**: Interactive Python-based visualization
- **3D Globe Visualization**: Real-time satellite orbit tracking

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
├── web/                       # Next.js Web Application
│   ├── public/
│   │   ├── model.onnx        # ONNX model for browser inference
│   │   └── telemetry_data.json
│   ├── src/
│   │   ├── app/              # Next.js App Router
│   │   ├── components/       # React components (Globe, Oscilloscope, etc.)
│   │   ├── hooks/            # Custom hooks (useInference)
│   │   ├── lib/              # ONNX session management
│   │   └── store/            # Zustand state management
│   ├── prepare_assets.py     # PyTorch → ONNX conversion script
│   └── package.json
├── streamlit_app.py          # Streamlit dashboard
├── best_model.pth            # Trained model checkpoint
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

### Streamlit Dashboard

Launch the interactive Python dashboard:

```bash
streamlit run streamlit_app.py
```

Features:
- Real-time telemetry visualization
- Anomaly detection with confidence scores
- Interactive controls (play/pause, speed, sabotage simulation)
- 3D satellite orbit visualization

### Web Dashboard (Next.js + ONNX)

The web application runs AI inference directly in the browser using ONNX Runtime WebAssembly.

#### 1. Convert model to ONNX format

```bash
cd web
python prepare_assets.py
```

This creates:
- `public/model.onnx` - Model for browser inference
- `public/telemetry_data.json` - Telemetry data

> **Note**: If you encounter `onnxscript` errors with PyTorch 2.x, use Kaggle/Colab to run the conversion (see troubleshooting section).

#### 2. Install and run

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000 to view the Space Command Center dashboard.

#### Web Dashboard Features

- **3D Globe**: Real-time satellite position with orbit path
- **Oscilloscope**: Live telemetry waveforms (8 channels)
- **HUD Status**: Anomaly probability with color-coded alerts
- **Control Panel**: Playback controls, speed adjustment, sabotage simulation
- **Mission Log**: Real-time event logging with timestamps

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

## Troubleshooting

### PyTorch 2.x ONNX Export Error

If you encounter this error when running `prepare_assets.py`:

```
ModuleNotFoundError: No module named 'onnxscript'
```

**Cause**: PyTorch 2.x requires `onnxscript` for ONNX export, which has heavy dependencies.

**Solutions**:

1. **Use Kaggle/Google Colab** (recommended):
   - Upload `best_model.pth` to Kaggle
   - Run the conversion notebook (see `web/kaggle_convert.py`)
   - Download the generated `model.onnx`

2. **Install onnxscript** (requires ~500MB):
   ```bash
   pip install onnxscript onnx
   ```

3. **Downgrade PyTorch** to 1.x:
   ```bash
   pip install torch==1.13.1
   ```

### Disk Space Issues

If `npm install` fails with `ENOSPC`:

```bash
# Clean npm cache
npm cache clean --force

# Use minimal install
npm install --prefer-offline --no-audit
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Training | PyTorch, Hydra, WandB |
| Model Export | ONNX, TorchScript |
| Web Frontend | Next.js 14, React, TypeScript |
| Browser Inference | ONNX Runtime Web (WASM) |
| State Management | Zustand |
| Visualization | react-globe.gl, Recharts |
| Styling | Tailwind CSS |
| Python Dashboard | Streamlit |

## References

- [NASA Telemanom](https://github.com/khundman/telemanom)
- [Detecting Spacecraft Anomalies Using LSTMs](https://arxiv.org/abs/1802.04431)
- [Hydra Documentation](https://hydra.cc/)
- [Weights & Biases](https://wandb.ai/)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [Next.js Documentation](https://nextjs.org/docs)
