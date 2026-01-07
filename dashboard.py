# =============================================================================
# Tiny-Sat-Anomaly: Real-Time Mission Control Dashboard
# =============================================================================
"""
Streamlit dashboard for real-time satellite telemetry anomaly detection.
Simulates a mission control room with live signal monitoring.

Usage:
    streamlit run dashboard.py
"""

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_model import LSTMAnomalyDetector

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Tiny-Sat Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - DARK SCI-FI THEME
# =============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
        border-right: 2px solid #00d4ff;
    }

    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        text-shadow: 0 0 10px #00d4ff40;
        font-family: 'Courier New', monospace;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-family: 'Courier New', monospace;
    }

    /* Status badges */
    .status-nominal {
        background: linear-gradient(90deg, #00ff8820, #00ff8810);
        border: 2px solid #00ff88;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px #00ff8840;
    }

    .status-critical {
        background: linear-gradient(90deg, #ff000020, #ff000010);
        border: 2px solid #ff0000;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px #ff000040;
        animation: pulse 1s infinite;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px #ff000040; }
        50% { box-shadow: 0 0 40px #ff000080; }
    }

    .status-text {
        font-size: 1.5rem;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }

    .nominal-text { color: #00ff88; }
    .critical-text { color: #ff0000; }

    /* Data panel */
    .data-panel {
        background: #1a1a2e;
        border: 1px solid #00d4ff40;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Terminal style text */
    .terminal {
        font-family: 'Courier New', monospace;
        color: #00ff88;
        background: #0a0a0f;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #00ff8840;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chart container */
    .chart-container {
        border: 1px solid #00d4ff40;
        border-radius: 10px;
        padding: 10px;
        background: #0a0a0f;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED RESOURCE LOADING
# =============================================================================
@st.cache_resource
def load_model() -> LSTMAnomalyDetector:
    """Load the trained LSTM model."""
    model = LSTMAnomalyDetector(
        input_size=25,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.2,
        bidirectional=False
    )

    # Load trained weights
    checkpoint = torch.load(
        'best_model.pth',
        map_location=torch.device('cpu'),
        weights_only=False
    )

    # Handle both direct state_dict and checkpoint format
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


@st.cache_data
def load_telemetry_data() -> np.ndarray:
    """Load the S-1 telemetry data."""
    data = np.load('S-1.npy')
    return data


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_anomaly(model: LSTMAnomalyDetector, sequence: np.ndarray) -> tuple:
    """
    Predict anomaly probability for a sequence.

    Args:
        model: Trained LSTM model
        sequence: Array of shape (window_size, n_features)

    Returns:
        Tuple of (anomaly_probability, predicted_class)
    """
    with torch.no_grad():
        # Prepare input: (1, window_size, n_features)
        x = torch.FloatTensor(sequence).unsqueeze(0)

        # Get prediction
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=1)

        # Anomaly probability (class 1)
        anomaly_prob = probs[0, 1].item()
        predicted_class = torch.argmax(probs, dim=1).item()

    return anomaly_prob, predicted_class


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>🛰️ TINY-SAT MISSION CONTROL</h1>
        <p style="color: #00d4ff; font-family: 'Courier New', monospace;">
            Real-Time Telemetry Anomaly Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # SIDEBAR CONTROLS
    # =========================================================================
    with st.sidebar:
        st.markdown("## ⚙️ MISSION CONTROLS")
        st.markdown("---")

        # Start/Stop button
        if 'mission_active' not in st.session_state:
            st.session_state.mission_active = False

        if st.button("🚀 START MISSION" if not st.session_state.mission_active else "⏹️ ABORT MISSION",
                     use_container_width=True):
            st.session_state.mission_active = not st.session_state.mission_active
            if st.session_state.mission_active:
                st.session_state.data_index = 0
                st.session_state.buffer = deque(maxlen=50)
                st.session_state.history = deque(maxlen=200)
                st.session_state.prob_history = deque(maxlen=200)
                st.session_state.anomaly_count = 0

        st.markdown("---")

        # Simulation speed
        speed = st.slider(
            "⏱️ Simulation Speed",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Delay between data points (seconds)"
        )

        # Threshold
        threshold = st.slider(
            "🎯 Anomaly Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for anomaly detection"
        )

        st.markdown("---")

        # Mission status
        st.markdown("### 📊 MISSION STATUS")
        if st.session_state.get('mission_active', False):
            st.markdown('<p class="terminal">STATUS: ACTIVE</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="terminal">STATUS: STANDBY</p>', unsafe_allow_html=True)

    # =========================================================================
    # LOAD RESOURCES
    # =========================================================================
    try:
        model = load_model()
        telemetry_data = load_telemetry_data()
        st.sidebar.success(f"✅ Model loaded")
        st.sidebar.success(f"✅ Data loaded: {telemetry_data.shape}")
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Make sure `best_model.pth` and `S-1.npy` are in the project root.")
        return
    except Exception as e:
        st.error(f"❌ Error loading resources: {e}")
        return

    # =========================================================================
    # MAIN DISPLAY AREA
    # =========================================================================
    # Top row: KPIs
    col1, col2, col3, col4 = st.columns(4)

    # Placeholders for real-time updates
    prob_placeholder = col1.empty()
    status_placeholder = col2.empty()
    timestep_placeholder = col3.empty()
    anomaly_count_placeholder = col4.empty()

    # Chart placeholder
    st.markdown("### 📈 LIVE TELEMETRY FEED")
    chart_placeholder = st.empty()

    # Secondary chart for anomaly probability
    st.markdown("### 🎯 ANOMALY PROBABILITY")
    prob_chart_placeholder = st.empty()

    # =========================================================================
    # SIMULATION LOOP
    # =========================================================================
    if st.session_state.get('mission_active', False):
        # Initialize if needed
        if 'data_index' not in st.session_state:
            st.session_state.data_index = 0
            st.session_state.buffer = deque(maxlen=50)
            st.session_state.history = deque(maxlen=200)
            st.session_state.prob_history = deque(maxlen=200)
            st.session_state.anomaly_count = 0

        # Process data points
        while st.session_state.mission_active and st.session_state.data_index < len(telemetry_data):
            idx = st.session_state.data_index
            current_data = telemetry_data[idx]

            # Add to buffer
            st.session_state.buffer.append(current_data)

            # Add channel 0 to history for visualization
            st.session_state.history.append(current_data[0] if len(current_data.shape) > 0 else current_data)

            # Make prediction if we have enough data
            if len(st.session_state.buffer) >= 50:
                sequence = np.array(list(st.session_state.buffer))
                anomaly_prob, pred_class = predict_anomaly(model, sequence)

                if anomaly_prob > threshold:
                    st.session_state.anomaly_count += 1
            else:
                anomaly_prob = 0.0
                pred_class = 0

            # Store probability history
            st.session_state.prob_history.append(anomaly_prob * 100)

            # Update KPIs
            prob_placeholder.metric(
                "🎯 ANOMALY PROBABILITY",
                f"{anomaly_prob * 100:.1f}%",
                delta=f"{(anomaly_prob - 0.5) * 100:+.1f}%" if len(st.session_state.buffer) >= 50 else None
            )

            # Status badge
            if anomaly_prob > threshold:
                status_placeholder.markdown("""
                <div class="status-critical">
                    <span class="status-text critical-text">🔴 CRITICAL ANOMALY DETECTED</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                status_placeholder.markdown("""
                <div class="status-nominal">
                    <span class="status-text nominal-text">🟢 NOMINAL</span>
                </div>
                """, unsafe_allow_html=True)

            # Timestep counter
            timestep_placeholder.metric(
                "⏱️ TIMESTEP",
                f"{idx:,} / {len(telemetry_data):,}",
                delta=f"{idx / len(telemetry_data) * 100:.1f}%"
            )

            # Anomaly counter
            anomaly_count_placeholder.metric(
                "⚠️ ANOMALIES DETECTED",
                st.session_state.anomaly_count,
                delta="+1" if anomaly_prob > threshold else None,
                delta_color="inverse"
            )

            # Update telemetry chart
            if len(st.session_state.history) > 0:
                chart_data = pd.DataFrame({
                    'Signal (Channel 0)': list(st.session_state.history)
                })
                chart_placeholder.line_chart(
                    chart_data,
                    use_container_width=True,
                    height=300
                )

            # Update probability chart
            if len(st.session_state.prob_history) > 0:
                prob_data = pd.DataFrame({
                    'Anomaly Probability (%)': list(st.session_state.prob_history),
                    'Threshold': [threshold * 100] * len(st.session_state.prob_history)
                })
                prob_chart_placeholder.line_chart(
                    prob_data,
                    use_container_width=True,
                    height=200
                )

            # Increment and wait
            st.session_state.data_index += 1
            time.sleep(speed)

        # Mission complete
        if st.session_state.data_index >= len(telemetry_data):
            st.session_state.mission_active = False
            st.balloons()
            st.success("🎉 MISSION COMPLETE - All telemetry data processed!")

    else:
        # Standby mode - show static display
        prob_placeholder.metric("🎯 ANOMALY PROBABILITY", "-- %")
        status_placeholder.markdown("""
        <div class="status-nominal">
            <span class="status-text nominal-text">⏸️ STANDBY</span>
        </div>
        """, unsafe_allow_html=True)
        timestep_placeholder.metric("⏱️ TIMESTEP", f"0 / {len(telemetry_data):,}")
        anomaly_count_placeholder.metric("⚠️ ANOMALIES DETECTED", 0)

        # Show sample data preview
        st.markdown("### 📊 TELEMETRY DATA PREVIEW")
        preview_df = pd.DataFrame(
            telemetry_data[:100, :5],
            columns=[f"Channel {i}" for i in range(5)]
        )
        st.line_chart(preview_df, use_container_width=True, height=300)

        st.info("👆 Click **START MISSION** in the sidebar to begin real-time monitoring.")

    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-family: 'Courier New', monospace;">
        <p>TINY-SAT-ANOMALY v1.0 | LSTM Anomaly Detection | NASA Telemanom Dataset</p>
        <p>🛰️ Satellite Telemetry Monitoring System 🛰️</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
