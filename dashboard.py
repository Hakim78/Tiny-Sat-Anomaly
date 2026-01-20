# =============================================================================
# TINY-SAT ANOMALY DETECTION - MISSION CONTROL CENTER
# =============================================================================
"""
Professional SSA (Space Situational Awareness) Dashboard
Real-time satellite telemetry monitoring with 3D orbital visualization.

Features:
- Live telemetry replay from NASA SMAP data
- LSTM-based anomaly detection
- 3D Globe visualization with Plotly (no API key required)
- Solar Flare injection for chaos testing
- Cyberpunk HUD-style dark theme interface

Usage:
    streamlit run dashboard.py

Requirements:
    - best_model.pth (trained LSTM model)
    - S-1.npy (NASA telemetry data)

Author: MLOps Team
Version: 2.1.0
"""

import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_model import LSTMAnomalyDetector

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="TINY-SAT Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "TINY-SAT Anomaly Detection System v2.0"
    }
)

# =============================================================================
# CUSTOM CSS - CYBERPUNK SPACE HUD THEME
# =============================================================================
st.markdown("""
<style>
    /* ===== GLOBAL DARK THEME ===== */
    .stApp {
        background: radial-gradient(ellipse at top, #0a0e14 0%, #0d1117 50%, #080b0f 100%);
        color: #c9d1d9;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080b0f 0%, #0d1117 50%, #1a2332 100%);
        border-right: 2px solid #30363d;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5);
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #238636 0%, #2ea043 50%, #238636 100%);
        color: white;
        border: 1px solid #3fb950;
        border-radius: 8px;
        padding: 14px;
        font-weight: 700;
        font-family: 'SF Mono', 'Consolas', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px #23863640;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 50%, #2ea043 100%);
        box-shadow: 0 0 30px #3fb95080;
        transform: translateY(-2px);
    }

    /* ===== HUD METRIC CARDS ===== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a2332 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(88, 166, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #58a6ff, transparent);
    }

    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.8rem !important;
        font-family: 'SF Mono', 'Consolas', monospace !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 1.8rem !important;
        font-family: 'SF Mono', 'Consolas', monospace !important;
        text-shadow: 0 0 20px #58a6ff60;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'SF Mono', 'Consolas', monospace !important;
    }

    /* ===== STATUS BADGES ===== */
    .status-nominal {
        background: linear-gradient(135deg, #0a1a0a 0%, #0d2818 50%, #0a1a0a 100%);
        border: 2px solid #3fb950;
        border-radius: 16px;
        padding: 28px 20px;
        text-align: center;
        box-shadow: 0 0 40px #3fb95040, inset 0 0 30px #3fb95015;
        animation: glow-green 2s ease-in-out infinite alternate;
        position: relative;
        overflow: hidden;
    }

    .status-nominal::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, #3fb95010 0%, transparent 70%);
        animation: pulse-bg 3s ease-in-out infinite;
    }

    .status-critical {
        background: linear-gradient(135deg, #1a0505 0%, #2d0a0a 50%, #1a0505 100%);
        border: 2px solid #f85149;
        border-radius: 16px;
        padding: 28px 20px;
        text-align: center;
        box-shadow: 0 0 50px #f8514950, inset 0 0 30px #f8514920;
        animation: glow-red 0.4s ease-in-out infinite alternate;
        position: relative;
    }

    .status-standby {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 2px solid #484f58;
        border-radius: 16px;
        padding: 28px 20px;
        text-align: center;
        box-shadow: 0 0 20px #8b949e15;
    }

    @keyframes glow-green {
        from { box-shadow: 0 0 30px #3fb95030, inset 0 0 20px #3fb95010; }
        to { box-shadow: 0 0 60px #3fb95060, inset 0 0 40px #3fb95020; }
    }

    @keyframes glow-red {
        from { box-shadow: 0 0 30px #f8514940, inset 0 0 20px #f8514915; }
        to { box-shadow: 0 0 80px #f8514980, inset 0 0 50px #f8514930; }
    }

    @keyframes pulse-bg {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }

    .status-text {
        font-size: 1.3rem;
        font-weight: 800;
        font-family: 'SF Mono', 'Consolas', monospace;
        letter-spacing: 3px;
        position: relative;
        z-index: 1;
    }

    .nominal-text { color: #3fb950; text-shadow: 0 0 20px #3fb950, 0 0 40px #3fb95080; }
    .critical-text { color: #f85149; text-shadow: 0 0 20px #f85149, 0 0 40px #f8514980; }
    .standby-text { color: #8b949e; text-shadow: 0 0 10px #8b949e40; }

    /* ===== PANEL STYLING ===== */
    .hud-panel {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .panel-title {
        color: #58a6ff;
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #30363d;
        text-shadow: 0 0 10px #58a6ff40;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .panel-title::before {
        content: '▸';
        color: #3fb950;
    }

    /* ===== LOG TERMINAL ===== */
    .log-terminal {
        background: linear-gradient(180deg, #080b0f 0%, #0d1117 100%);
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
        font-family: 'SF Mono', 'Consolas', 'Courier New', monospace;
        font-size: 0.8rem;
        color: #8b949e;
        max-height: 180px;
        overflow-y: auto;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
    }

    .log-terminal::-webkit-scrollbar {
        width: 6px;
    }

    .log-terminal::-webkit-scrollbar-track {
        background: #0d1117;
    }

    .log-terminal::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 3px;
    }

    .log-entry {
        margin: 6px 0;
        padding: 4px 0;
        border-bottom: 1px solid #21262d20;
    }
    .log-time { color: #484f58; }
    .log-info { color: #58a6ff; font-weight: 600; }
    .log-warn { color: #d29922; font-weight: 600; }
    .log-error { color: #f85149; font-weight: 700; text-shadow: 0 0 5px #f8514950; }
    .log-success { color: #3fb950; font-weight: 600; }

    /* ===== HEADER ===== */
    .mission-header {
        text-align: center;
        padding: 30px 0;
        border-bottom: 2px solid #21262d;
        margin-bottom: 25px;
        background: linear-gradient(180deg, transparent 0%, #0d111710 100%);
        position: relative;
    }

    .mission-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #58a6ff, transparent);
    }

    .mission-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #58a6ff;
        font-family: 'SF Mono', 'Consolas', monospace;
        text-shadow: 0 0 40px #58a6ff60, 0 0 80px #58a6ff30;
        letter-spacing: 6px;
        margin: 0;
    }

    .mission-subtitle {
        color: #6e7681;
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.85rem;
        margin-top: 12px;
        letter-spacing: 4px;
        text-transform: uppercase;
    }

    /* ===== SCANLINE EFFECT (optional) ===== */
    .mission-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 0, 0, 0.1) 2px,
            rgba(0, 0, 0, 0.1) 4px
        );
        pointer-events: none;
        opacity: 0.3;
    }

    /* ===== CHARTS CONTAINER ===== */
    .stPlotlyChart {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED RESOURCE LOADING
# =============================================================================
@st.cache_resource
def load_model() -> Tuple[Optional[LSTMAnomalyDetector], str]:
    """
    Load the trained LSTM anomaly detection model.

    Returns:
        Tuple of (model or None, status_message)
    """
    try:
        model = LSTMAnomalyDetector(
            input_size=25,
            hidden_size=128,
            num_layers=2,
            num_classes=2,
            dropout=0.2,
            bidirectional=False
        )

        # Load with CPU mapping and allow pickle (PyTorch 2.6+ compatibility)
        checkpoint = torch.load(
            'best_model.pth',
            map_location=torch.device('cpu'),
            weights_only=False
        )

        # Handle checkpoint format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'N/A')
            status = f"✅ Model loaded (Epoch {epoch})"
        else:
            model.load_state_dict(checkpoint)
            status = "✅ Model loaded"

        model.eval()
        return model, status

    except FileNotFoundError:
        return None, "⚠️ best_model.pth not found"
    except Exception as e:
        return None, f"❌ Model error: {str(e)[:50]}"


@st.cache_data
def load_telemetry_data() -> Tuple[Optional[np.ndarray], str]:
    """
    Load NASA telemetry data with fallback to synthetic data.

    Returns:
        Tuple of (data array, status_message)
    """
    try:
        data = np.load('S-1.npy')
        return data, f"✅ S-1.npy loaded ({data.shape[0]:,} samples)"
    except FileNotFoundError:
        # Generate fallback synthetic data
        np.random.seed(42)
        t = np.linspace(0, 100 * np.pi, 5000)
        synthetic = np.column_stack([
            np.sin(t * (i + 1) / 10) + np.random.normal(0, 0.1, len(t))
            for i in range(25)
        ])
        return synthetic, "⚠️ Using synthetic fallback data"
    except Exception as e:
        return None, f"❌ Data error: {str(e)[:50]}"


# =============================================================================
# PREDICTION & ORBIT FUNCTIONS
# =============================================================================
def predict_anomaly(model: LSTMAnomalyDetector, sequence: np.ndarray) -> Tuple[float, int]:
    """
    Run inference on a sequence.

    Args:
        model: Trained LSTM model
        sequence: Array of shape (50, 25)

    Returns:
        Tuple of (anomaly_probability, predicted_class)
    """
    with torch.no_grad():
        x = torch.FloatTensor(sequence).unsqueeze(0)
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=1)
        anomaly_prob = probs[0, 1].item()
        pred_class = torch.argmax(probs, dim=1).item()
    return anomaly_prob, pred_class


def calculate_orbit_position(t: int) -> Tuple[float, float, float]:
    """
    Calculate satellite position for polar orbit simulation.

    Args:
        t: Time step index

    Returns:
        Tuple of (latitude, longitude, altitude_km)
    """
    # Polar orbit simulation
    lat = np.sin(t * 0.02) * 75  # Oscillate between ±75°
    lon = ((t * 0.8) % 360) - 180  # Rotate around Earth
    alt = 685 + np.sin(t * 0.1) * 20  # ~685km altitude with variation
    return float(lat), float(lon), float(alt)


def create_globe_view(
    lat: float,
    lon: float,
    alt: float,
    is_anomaly: bool,
    orbit_trail: List[Tuple[float, float]]
) -> go.Figure:
    """
    Create Plotly 3D Globe visualization with satellite and orbit trail.

    Args:
        lat: Satellite latitude
        lon: Satellite longitude
        alt: Satellite altitude in km
        is_anomaly: Whether current state is anomalous
        orbit_trail: List of (lon, lat) tuples for orbit path

    Returns:
        Plotly Figure object with 3D globe
    """
    fig = go.Figure()

    # Satellite marker color based on status
    sat_color = '#f85149' if is_anomaly else '#3fb950'
    sat_size = 18 if is_anomaly else 14

    # Add orbit trail (ground track)
    if len(orbit_trail) > 1:
        trail_lons = [p[0] for p in orbit_trail]
        trail_lats = [p[1] for p in orbit_trail]

        fig.add_trace(go.Scattergeo(
            lon=trail_lons,
            lat=trail_lats,
            mode='lines',
            line=dict(width=2, color='#58a6ff'),
            opacity=0.6,
            name='Orbit Track',
            hoverinfo='skip'
        ))

    # Add satellite marker
    fig.add_trace(go.Scattergeo(
        lon=[lon],
        lat=[lat],
        mode='markers+text',
        marker=dict(
            size=sat_size,
            color=sat_color,
            symbol='diamond',
            line=dict(width=2, color='white')
        ),
        text=['🛰️'],
        textposition='top center',
        textfont=dict(size=16),
        name='SMAP Satellite',
        hovertemplate=(
            '<b>SMAP SATELLITE</b><br>'
            f'Lat: {lat:.2f}°<br>'
            f'Lon: {lon:.2f}°<br>'
            f'Alt: {alt:.0f} km<br>'
            '<extra></extra>'
        )
    ))

    # Configure 3D orthographic globe projection
    fig.update_geos(
        projection_type='orthographic',
        projection_rotation=dict(lon=lon, lat=lat/2),  # Center on satellite
        showland=True,
        landcolor='#1a2332',
        showocean=True,
        oceancolor='#0d1117',
        showlakes=True,
        lakecolor='#161b22',
        showrivers=False,
        showcountries=True,
        countrycolor='#30363d',
        countrywidth=0.5,
        showcoastlines=True,
        coastlinecolor='#58a6ff',
        coastlinewidth=1,
        showframe=False,
        bgcolor='rgba(0,0,0,0)',
        lonaxis=dict(showgrid=True, gridcolor='#21262d', gridwidth=0.5),
        lataxis=dict(showgrid=True, gridcolor='#21262d', gridwidth=0.5),
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=320,
        showlegend=False,
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
        )
    )

    return fig


def create_telemetry_chart(history: deque, anomaly_indices: list) -> go.Figure:
    """
    Create Plotly telemetry chart with anomaly markers.

    Args:
        history: Deque of signal values
        anomaly_indices: List of indices where anomalies occurred

    Returns:
        Plotly Figure object
    """
    if len(history) == 0:
        return go.Figure()

    x = list(range(len(history)))
    y = list(history)

    fig = go.Figure()

    # Main signal trace with gradient effect
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='Channel 0',
        line=dict(color='#58a6ff', width=2.5, shape='spline', smoothing=0.8),
        fill='tozeroy',
        fillcolor='rgba(88, 166, 255, 0.15)',
        hovertemplate='<b>Step:</b> %{x}<br><b>Value:</b> %{y:.4f}<extra></extra>'
    ))

    # Anomaly markers with glow effect
    if anomaly_indices:
        anomaly_x = [i for i in anomaly_indices if i < len(history)]
        anomaly_y = [history[i] for i in anomaly_x if i < len(history)]

        # Outer glow
        fig.add_trace(go.Scatter(
            x=anomaly_x, y=anomaly_y,
            mode='markers',
            name='',
            showlegend=False,
            marker=dict(
                color='rgba(248, 81, 73, 0.3)',
                size=24,
                symbol='circle'
            ),
            hoverinfo='skip'
        ))

        # Inner marker
        fig.add_trace(go.Scatter(
            x=anomaly_x, y=anomaly_y,
            mode='markers',
            name='ANOMALY',
            marker=dict(
                color='#f85149',
                size=10,
                symbol='diamond',
                line=dict(width=2, color='#ffffff')
            ),
            hovertemplate='<b>⚠️ ANOMALY DETECTED</b><br>Step: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(8, 11, 15, 0.9)',
        font=dict(family="SF Mono, Consolas, monospace", color='#8b949e', size=11),
        margin=dict(l=60, r=30, t=40, b=50),
        height=350,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10)
        ),
        xaxis=dict(
            title=dict(text="TIME STEPS", font=dict(size=10, color='#6e7681')),
            gridcolor='#21262d',
            zerolinecolor='#30363d',
            tickfont=dict(size=9),
            showspikes=True,
            spikecolor='#58a6ff',
            spikethickness=1,
        ),
        yaxis=dict(
            title=dict(text="SIGNAL AMPLITUDE", font=dict(size=10, color='#6e7681')),
            gridcolor='#21262d',
            zerolinecolor='#30363d',
            tickfont=dict(size=9),
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            font=dict(family='SF Mono, Consolas, monospace', size=11)
        )
    )

    return fig


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main():
    """Main dashboard application."""

    # =========================================================================
    # HEADER
    # =========================================================================
    st.markdown("""
    <div class="mission-header">
        <h1 class="mission-title">🛰️ TINY-SAT MISSION CONTROL</h1>
        <p class="mission-subtitle">SPACE SITUATIONAL AWARENESS • ANOMALY DETECTION SYSTEM</p>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # LOAD RESOURCES
    # =========================================================================
    model, model_status = load_model()
    telemetry_data, data_status = load_telemetry_data()

    # =========================================================================
    # SIDEBAR CONTROLS
    # =========================================================================
    with st.sidebar:
        st.markdown("## ⚙️ MISSION CONTROL")
        st.markdown("---")

        # Resource status
        st.markdown("### 📦 RESOURCES")
        st.markdown(f"<small>{model_status}</small>", unsafe_allow_html=True)
        st.markdown(f"<small>{data_status}</small>", unsafe_allow_html=True)
        st.markdown("---")

        # Initialize session state
        if 'simulation_active' not in st.session_state:
            st.session_state.simulation_active = False
            st.session_state.step = 0
            st.session_state.buffer = deque(maxlen=50)
            st.session_state.signal_history = deque(maxlen=200)
            st.session_state.prob_history = deque(maxlen=200)
            st.session_state.anomaly_count = 0
            st.session_state.anomaly_indices = []
            st.session_state.logs = deque(maxlen=20)
            st.session_state.solar_flare_countdown = 0
            st.session_state.orbit_trail = deque(maxlen=50)

        # Start/Stop button
        if model is not None and telemetry_data is not None:
            btn_text = "⏹️ ABORT MISSION" if st.session_state.simulation_active else "🚀 START SIMULATION"
            if st.button(btn_text, use_container_width=True, type="primary"):
                st.session_state.simulation_active = not st.session_state.simulation_active
                if st.session_state.simulation_active:
                    # Reset state
                    st.session_state.step = 0
                    st.session_state.buffer = deque(maxlen=50)
                    st.session_state.signal_history = deque(maxlen=200)
                    st.session_state.prob_history = deque(maxlen=200)
                    st.session_state.anomaly_count = 0
                    st.session_state.anomaly_indices = []
                    st.session_state.logs = deque(maxlen=20)
                    st.session_state.orbit_trail = deque(maxlen=50)
                    st.session_state.logs.append(("INFO", "Mission started"))
        else:
            st.button("🚀 START SIMULATION", disabled=True, use_container_width=True)
            st.warning("Load model and data first")

        st.markdown("---")

        # Simulation controls
        st.markdown("### ⚡ SIMULATION")

        speed = st.slider(
            "Refresh Rate (ms)",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Lower = faster simulation"
        )

        threshold = st.slider(
            "Anomaly Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

        st.markdown("---")

        # Chaos Monkey - Solar Flare
        st.markdown("### 🔥 CHAOS ENGINEERING")
        if st.button("⚡ TRIGGER SOLAR FLARE", use_container_width=True):
            st.session_state.solar_flare_countdown = 10
            st.session_state.logs.append(("WARN", "SOLAR FLARE INCOMING!"))
            st.toast("☀️ Solar Flare triggered!", icon="⚡")

        if st.session_state.solar_flare_countdown > 0:
            st.warning(f"🔥 Flare active: {st.session_state.solar_flare_countdown} cycles")

        st.markdown("---")

        # Mission stats
        st.markdown("### 📊 MISSION STATS")
        if telemetry_data is not None:
            progress = st.session_state.step / len(telemetry_data) if len(telemetry_data) > 0 else 0
            st.progress(progress, text=f"{progress*100:.1f}% Complete")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Packets", f"{st.session_state.step:,}")
        with col2:
            st.metric("Anomalies", st.session_state.anomaly_count)

    # =========================================================================
    # MAIN CONTENT AREA
    # =========================================================================

    # Top row: KPI metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🛰️ SATELLITE",
            "SMAP",
            "NASA JPL"
        )

    prob_metric = col2.empty()
    status_badge = col3.empty()
    orbit_metric = col4.empty()

    st.markdown("---")

    # Middle row: Main displays
    col_chart, col_globe = st.columns([2, 1])

    with col_chart:
        st.markdown('<div class="panel-title">📡 LIVE TELEMETRY FEED</div>', unsafe_allow_html=True)
        chart_placeholder = st.empty()

    with col_globe:
        st.markdown('<div class="panel-title">🌍 ORBITAL POSITION</div>', unsafe_allow_html=True)
        globe_placeholder = st.empty()

    # Bottom row: Logs
    st.markdown("---")
    st.markdown('<div class="panel-title">📋 MISSION LOG</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()

    # =========================================================================
    # SIMULATION LOOP
    # =========================================================================
    if st.session_state.simulation_active and model is not None and telemetry_data is not None:

        while st.session_state.simulation_active and st.session_state.step < len(telemetry_data):
            step = st.session_state.step
            current_data = telemetry_data[step].copy()

            # ===== SOLAR FLARE INJECTION =====
            if st.session_state.solar_flare_countdown > 0:
                # Inject massive spike
                current_data = current_data + np.random.uniform(3.0, 5.0, size=current_data.shape)
                st.session_state.solar_flare_countdown -= 1
                if st.session_state.solar_flare_countdown == 0:
                    st.session_state.logs.append(("SUCCESS", "Solar flare subsided"))

            # Add to buffer
            st.session_state.buffer.append(current_data)

            # Signal history (Channel 0)
            signal_val = current_data[0] if len(current_data.shape) > 0 else current_data
            st.session_state.signal_history.append(signal_val)

            # ===== PREDICTION =====
            if len(st.session_state.buffer) >= 50:
                sequence = np.array(list(st.session_state.buffer))
                anomaly_prob, pred_class = predict_anomaly(model, sequence)
                is_anomaly = anomaly_prob > threshold

                if is_anomaly:
                    st.session_state.anomaly_count += 1
                    st.session_state.anomaly_indices.append(len(st.session_state.signal_history) - 1)
                    st.session_state.logs.append(("ERROR", f"ANOMALY @ step {step} (P={anomaly_prob:.2%})"))
            else:
                anomaly_prob = 0.0
                is_anomaly = False

            st.session_state.prob_history.append(anomaly_prob * 100)

            # ===== CALCULATE ORBIT =====
            lat, lon, alt = calculate_orbit_position(step)

            # Update orbit trail
            st.session_state.orbit_trail.append((lon, lat))

            # ===== UPDATE DISPLAYS =====

            # Probability metric
            delta = f"{(anomaly_prob - 0.5) * 100:+.1f}%" if len(st.session_state.buffer) >= 50 else None
            prob_metric.metric(
                "🎯 ANOMALY PROB",
                f"{anomaly_prob * 100:.1f}%",
                delta=delta,
                delta_color="inverse" if anomaly_prob > threshold else "normal"
            )

            # Status badge
            if not st.session_state.simulation_active:
                status_html = """
                <div class="status-standby">
                    <span class="status-text standby-text">⏸️ STANDBY</span>
                </div>
                """
            elif is_anomaly:
                status_html = """
                <div class="status-critical">
                    <span class="status-text critical-text">🔴 CRITICAL</span>
                </div>
                """
            else:
                status_html = """
                <div class="status-nominal">
                    <span class="status-text nominal-text">🟢 NOMINAL</span>
                </div>
                """
            status_badge.markdown(status_html, unsafe_allow_html=True)

            # Orbit info
            orbit_metric.metric(
                "🌍 ALTITUDE",
                f"{alt:.0f} km",
                f"Lat: {lat:.1f}° Lon: {lon:.1f}°"
            )

            # Telemetry chart
            fig = create_telemetry_chart(
                st.session_state.signal_history,
                st.session_state.anomaly_indices
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{step}")

            # Globe view
            orbit_trail_list = list(st.session_state.orbit_trail)
            globe_fig = create_globe_view(lat, lon, alt, is_anomaly, orbit_trail_list)
            globe_placeholder.plotly_chart(globe_fig, use_container_width=True, key=f"globe_{step}")

            # Mission log
            log_html = '<div class="log-terminal">'
            for log_type, msg in reversed(list(st.session_state.logs)):
                timestamp = datetime.now().strftime("%H:%M:%S")
                css_class = {
                    "INFO": "log-info",
                    "WARN": "log-warn",
                    "ERROR": "log-error",
                    "SUCCESS": "log-success"
                }.get(log_type, "log-info")
                log_html += f'<div class="log-entry"><span class="log-time">[{timestamp}]</span> <span class="{css_class}">[{log_type}]</span> {msg}</div>'
            log_html += '</div>'
            log_placeholder.markdown(log_html, unsafe_allow_html=True)

            # Increment and sleep
            st.session_state.step += 1
            time.sleep(speed / 1000)

        # Mission complete
        if st.session_state.step >= len(telemetry_data):
            st.session_state.simulation_active = False
            st.balloons()
            st.success(f"🎉 MISSION COMPLETE! Processed {st.session_state.step:,} packets. Detected {st.session_state.anomaly_count} anomalies.")

    else:
        # ===== STANDBY MODE =====
        prob_metric.metric("🎯 ANOMALY PROB", "-- %")
        status_badge.markdown("""
        <div class="status-standby">
            <span class="status-text standby-text">⏸️ STANDBY</span>
        </div>
        """, unsafe_allow_html=True)
        orbit_metric.metric("🌍 ALTITUDE", "-- km", "Awaiting signal")

        # Preview chart
        if telemetry_data is not None:
            preview_fig = go.Figure()
            preview_data = telemetry_data[:200, 0]
            preview_fig.add_trace(go.Scatter(
                y=preview_data,
                mode='lines',
                name='Preview Signal',
                line=dict(color='#58a6ff', width=2, shape='spline', smoothing=0.8),
                fill='tozeroy',
                fillcolor='rgba(88, 166, 255, 0.12)'
            ))
            preview_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(8, 11, 15, 0.9)',
                font=dict(family="SF Mono, Consolas, monospace", color='#8b949e', size=11),
                margin=dict(l=60, r=30, t=50, b=50),
                height=350,
                title=dict(
                    text="📊 PREVIEW: First 200 Samples",
                    font=dict(size=12, color='#6e7681'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(text="SAMPLE INDEX", font=dict(size=10, color='#6e7681')),
                    gridcolor='#21262d',
                    tickfont=dict(size=9)
                ),
                yaxis=dict(
                    title=dict(text="SIGNAL VALUE", font=dict(size=10, color='#6e7681')),
                    gridcolor='#21262d',
                    tickfont=dict(size=9)
                ),
                showlegend=False
            )
            chart_placeholder.plotly_chart(preview_fig, use_container_width=True)

        # Static globe (centered on Atlantic)
        globe_fig = create_globe_view(20, -30, 685, False, [])
        globe_placeholder.plotly_chart(globe_fig, use_container_width=True)

        # Standby log
        log_placeholder.markdown("""
        <div class="log-terminal">
            <div class="log-entry"><span class="log-time">[--:--:--]</span> <span class="log-info">[SYS]</span> Mission Control initialized</div>
            <div class="log-entry"><span class="log-time">[--:--:--]</span> <span class="log-info">[SYS]</span> Awaiting START command...</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #484f58; font-family: 'SF Mono', Consolas, monospace; font-size: 0.75rem; padding: 20px 0;">
        <p style="margin: 4px 0; color: #6e7681;">TINY-SAT-ANOMALY v2.1 • LSTM Neural Network • NASA SMAP Telemetry</p>
        <p style="margin: 4px 0;">🛰️ Space Situational Awareness System • Streamlit + PyTorch + Plotly 🛰️</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
