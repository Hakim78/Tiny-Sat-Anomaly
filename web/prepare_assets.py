#!/usr/bin/env python3
# =============================================================================
# Asset Preparation Script for Next.js + ONNX Deployment
# =============================================================================
"""
Converts PyTorch model and numpy data to web-friendly formats.

Outputs:
    - public/model.onnx: ONNX model for browser inference
    - public/telemetry_data.json: Compressed telemetry data

Usage:
    python prepare_assets.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Workaround for PyTorch 2.x onnxscript dependency
# Create minimal fake module to satisfy import, then use legacy exporter
class _FakeOnnxScript:
    """Minimal fake onnxscript to bypass PyTorch 2.x import requirement."""
    def __getattr__(self, name):
        return _FakeOnnxScript()
    def __call__(self, *args, **kwargs):
        return _FakeOnnxScript()

sys.modules['onnxscript'] = _FakeOnnxScript()
sys.modules['onnxscript.function_libs'] = _FakeOnnxScript()
sys.modules['onnxscript.function_libs.torch_lib'] = _FakeOnnxScript()
sys.modules['onnxscript.onnx_opset'] = _FakeOnnxScript()
sys.modules['onnxscript.onnx_types'] = _FakeOnnxScript()

os.environ["TORCH_ONNX_USE_OLD_EXPORTER"] = "1"

import torch.onnx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model import LSTMAnomalyDetector


def convert_model_to_onnx(
    model_path: str,
    output_path: str,
    input_size: int = 25,
    hidden_size: int = 128,
    num_layers: int = 2,
    seq_length: int = 50,
    opset_version: int = 14
) -> None:
    """
    Convert PyTorch LSTM model to ONNX format.

    Args:
        model_path: Path to .pth checkpoint
        output_path: Output path for .onnx file
        input_size: Number of input features
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        seq_length: Sequence length (window size)
        opset_version: ONNX opset version (12+ for onnxruntime-web)
    """
    print("=" * 60)
    print("PHASE 1: PyTorch -> ONNX Conversion")
    print("=" * 60)

    # Initialize model with same architecture
    model = LSTMAnomalyDetector(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
        dropout=0.0,  # Disable dropout for inference
        bidirectional=False
    )

    # Load checkpoint
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Metrics: {checkpoint.get('metrics', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully!")

    # Create dummy input for tracing
    # Shape: (batch_size, seq_length, input_size)
    batch_size = 1
    dummy_input = torch.randn(batch_size, seq_length, input_size)

    # Initialize hidden states
    h0 = torch.zeros(num_layers, batch_size, hidden_size)
    c0 = torch.zeros(num_layers, batch_size, hidden_size)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Hidden state shape: h0={h0.shape}, c0={c0.shape}")

    # Create wrapper model that doesn't require hidden state input
    # This simplifies the ONNX graph for browser inference
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model

        def forward(self, x):
            # Initialize hidden state internally
            batch_size = x.size(0)
            device = x.device
            h0 = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size, device=device)
            c0 = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size, device=device)

            # Forward pass
            logits, _ = self.model(x, (h0, c0))

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            # Return anomaly probability (class 1)
            return probs[:, 1:2]  # Shape: (batch, 1)

    wrapper = ONNXWrapper(model)
    wrapper.eval()

    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Use torch.jit.trace to avoid onnxscript dependency in PyTorch 2.x
    # This creates a TorchScript model first, which exports cleanly to ONNX
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, dummy_input)

    torch.onnx.export(
        traced_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['anomaly_probability'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'anomaly_probability': {0: 'batch_size'}
        }
    )

    # Verify the exported model (optional - skip if onnx not installed)
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX model verification: PASSED")
    except ImportError:
        print("  [INFO] onnx package not installed - skipping verification")
    except Exception as e:
        print(f"  [WARNING] ONNX verification warning: {e}")

    file_size = Path(output_path).stat().st_size / 1024
    print(f"\n[SUCCESS] ONNX model saved: {output_path}")
    print(f"  File size: {file_size:.1f} KB")
    print(f"  Input: 'input' - shape (batch, {seq_length}, {input_size})")
    print(f"  Output: 'anomaly_probability' - shape (batch, 1)")


def convert_telemetry_to_json(
    npy_path: str,
    output_path: str,
    max_samples: int = 2500,
    precision: int = 4
) -> None:
    """
    Convert numpy telemetry data to JSON for browser.

    Args:
        npy_path: Path to .npy file
        output_path: Output path for .json file
        max_samples: Maximum number of samples to include
        precision: Decimal precision for float values
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Numpy -> JSON Conversion")
    print("=" * 60)

    print(f"Loading: {npy_path}")
    data = np.load(npy_path)

    print(f"  Original shape: {data.shape}")
    print(f"  Data type: {data.dtype}")

    # Limit samples for web demo
    if len(data) > max_samples:
        data = data[:max_samples]
        print(f"  Truncated to: {data.shape}")

    # Normalize data to [0, 1] range for consistent visualization
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1  # Avoid division by zero

    data_normalized = (data - data_min) / data_range

    # Round to reduce file size
    data_rounded = np.round(data_normalized, precision)

    # Convert to list of lists
    data_list = data_rounded.tolist()

    # Create output structure with metadata
    output = {
        "metadata": {
            "source": "NASA SMAP Satellite Telemetry",
            "samples": len(data_list),
            "features": len(data_list[0]) if data_list else 0,
            "normalized": True,
            "precision": precision
        },
        "data": data_list
    }

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))  # Compact format

    file_size = Path(output_path).stat().st_size / 1024
    print(f"\n[SUCCESS] JSON data saved: {output_path}")
    print(f"  File size: {file_size:.1f} KB")
    print(f"  Samples: {len(data_list)}")
    print(f"  Features: {output['metadata']['features']}")


def main():
    """Main execution."""
    print("\n" + "=" * 60)
    print("  TINY-SAT-ANOMALY: Web Asset Preparation")
    print("  PyTorch -> ONNX + NumPy -> JSON")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    web_public = Path(__file__).parent / "public"

    model_path = project_root / "best_model.pth"
    npy_path = project_root / "S-1.npy"

    onnx_output = web_public / "model.onnx"
    json_output = web_public / "telemetry_data.json"

    # Check inputs exist
    if not model_path.exists():
        # Try outputs directory
        alt_model_path = project_root / "outputs" / "checkpoints" / "best_model.pth"
        if alt_model_path.exists():
            model_path = alt_model_path
        else:
            print(f"[ERROR] Model not found: {model_path}")
            print("  Run training first: python src/train.py")
            return

    if not npy_path.exists():
        print(f"[WARNING] Telemetry data not found: {npy_path}")
        print("  Will generate synthetic data instead...")
        # Generate synthetic fallback
        np.random.seed(42)
        t = np.linspace(0, 100 * np.pi, 2500)
        synthetic = np.column_stack([
            np.sin(t * (i + 1) / 10) + np.random.normal(0, 0.1, len(t))
            for i in range(25)
        ])
        npy_path = project_root / "synthetic_data.npy"
        np.save(npy_path, synthetic)
        print(f"  Synthetic data saved: {npy_path}")

    # Convert model
    try:
        convert_model_to_onnx(
            model_path=str(model_path),
            output_path=str(onnx_output),
            input_size=25,
            hidden_size=128,
            num_layers=2,
            seq_length=50,
            opset_version=14
        )
    except Exception as e:
        print(f"[ERROR] Model conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Convert telemetry data
    try:
        convert_telemetry_to_json(
            npy_path=str(npy_path),
            output_path=str(json_output),
            max_samples=2500,
            precision=4
        )
    except Exception as e:
        print(f"[ERROR] Data conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("  ASSET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {onnx_output}")
    print(f"  - {json_output}")
    print("\nNext steps:")
    print("  1. cd web")
    print("  2. npm install")
    print("  3. npm run dev")


if __name__ == "__main__":
    main()
