# =============================================================================
# LSTM Anomaly Detector Model
# =============================================================================
"""
PyTorch LSTM architecture for satellite telemetry anomaly detection.

Implements a production-ready LSTM model with:
- Configurable architecture (layers, hidden size, bidirectional)
- Dropout regularization
- Layer normalization option
- Linear classification head
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class LSTMAnomalyDetector(nn.Module):
    """
    LSTM-based anomaly detector for time series data.

    Architecture:
    - LSTM encoder for sequence processing
    - Dropout for regularization
    - Linear classification head for binary classification

    This architecture is inspired by the Tiny Recursive Model concept,
    focusing on efficient sequence processing for telemetry data.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        batch_first: bool = True
    ) -> None:
        """
        Initialize the LSTM Anomaly Detector.

        Args:
            input_size: Number of features per timestep
            hidden_size: Dimension of LSTM hidden state
            num_layers: Number of stacked LSTM layers
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout probability (applied between LSTM layers)
            bidirectional: Use bidirectional LSTM
            batch_first: If True, input shape is (batch, seq, features)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        # Direction multiplier for output size
        self.num_directions = 2 if bidirectional else 1

        # Input projection layer (optional, for dimension matching)
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Layer normalization after input projection
        self.input_norm = nn.LayerNorm(hidden_size)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Post-LSTM dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Classification head
        classifier_input_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

        logger.info(f"LSTMAnomalyDetector initialized: {self}")

    def _init_weights(self) -> None:
        """
        Initialize model weights using Xavier/Glorot initialization.

        LSTM weights are initialized with orthogonal initialization
        for better gradient flow.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional initial hidden state tuple (h_0, c_0)

        Returns:
            Tuple of:
            - logits: Classification logits of shape (batch_size, num_classes)
            - hidden: Final hidden state tuple (h_n, c_n)
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # Input projection and normalization
        x = self.input_projection(x)
        x = self.input_norm(x)

        # LSTM encoding
        lstm_out, hidden = self.lstm(x, hidden)

        # Use the last timestep output for classification
        # For bidirectional: concatenate forward and backward final states
        if self.bidirectional:
            # lstm_out shape: (batch, seq_len, hidden_size * 2)
            final_output = lstm_out[:, -1, :]
        else:
            # lstm_out shape: (batch, seq_len, hidden_size)
            final_output = lstm_out[:, -1, :]

        # Apply dropout
        final_output = self.dropout_layer(final_output)

        # Classification
        logits = self.classifier(final_output)

        return logits, hidden

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """
        Initialize hidden state with zeros.

        Args:
            batch_size: Current batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h_0, c_0) initial hidden states
        """
        num_directions = self.num_directions
        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        return h_0, c_0

    def predict(self, x: Tensor) -> Tensor:
        """
        Make predictions (convenience method for inference).

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Predicted class labels of shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Get class probabilities (for threshold tuning).

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities

    def get_attention_weights(self, x: Tensor) -> Optional[Tensor]:
        """
        Get attention weights for interpretability (placeholder for extension).

        This method can be extended to add attention mechanisms
        for better interpretability of anomaly detection.

        Args:
            x: Input tensor

        Returns:
            None (to be implemented with attention mechanism)
        """
        # Placeholder for future attention implementation
        return None

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"LSTMAnomalyDetector(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_classes={self.num_classes},\n"
            f"  dropout={self.dropout},\n"
            f"  bidirectional={self.bidirectional}\n"
            f")"
        )


class TinyRecursiveLSTM(nn.Module):
    """
    Tiny Recursive LSTM variant with parameter sharing.

    This model uses weight sharing across time steps for a more
    memory-efficient implementation, suitable for edge deployment.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
        recursive_depth: int = 3
    ) -> None:
        """
        Initialize Tiny Recursive LSTM.

        Args:
            input_size: Number of input features
            hidden_size: Hidden state dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            recursive_depth: Number of recursive applications
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.recursive_depth = recursive_depth

        # Single LSTM cell (shared across recursive steps)
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        # Projection back to input space for recursion
        self.recursive_projection = nn.Linear(hidden_size, input_size)

        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.classifier = nn.Linear(hidden_size, num_classes)

        logger.info(f"TinyRecursiveLSTM initialized with depth {recursive_depth}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with recursive LSTM application.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]

            # Recursive refinement
            for _ in range(self.recursive_depth):
                h, c = self.lstm_cell(x_t, (h, c))
                x_t = self.recursive_projection(h)

        # Final classification
        h = self.dropout(h)
        logits = self.classifier(h)

        return logits


def create_model(cfg) -> nn.Module:
    """
    Factory function to create model from config.

    Args:
        cfg: Hydra configuration object

    Returns:
        Instantiated model
    """
    model = LSTMAnomalyDetector(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional
    )

    return model
