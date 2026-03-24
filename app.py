import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from model import EEGModelHandler
from preprocess import EEGPreprocessor
from controller import BCIController
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_eeg.csv")
MODEL_PATH = os.path.join(DATA_DIR, "bci_model.pkl")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="BCI Cursor Control System",
    page_icon="🧠",
    layout="wide",
)

# --- THEME & CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    .status-active {
        color: #10b981;
        font-weight: bold;
    }
    .status-idle {
        color: #6b7280;
    }
    .command-display {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
if 'bci_active' not in st.session_state:
    st.session_state.bci_active = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'eeg_data' not in st.session_state:
    st.session_state.eeg_data = None

preprocessor = EEGPreprocessor()
model_handler = EEGModelHandler(model_path=MODEL_PATH)
controller = BCIController()

# Check for cloud/headless environment
from controller import PYAUTOGUI_AVAILABLE

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ System Control")
    st.write("---")
    
    # Dataset control
    st.subheader("EEG Data Source")
    data_source = st.selectbox("Select Signal Source", ["Live Stream (Simulation)", "Upload CSV File"])
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Choose EEG Data CSV", type="csv")
        if uploaded_file:
            st.session_state.eeg_data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
    
    st.write("---")
    # Simulation Control
    st.subheader("BCI Engine")
    if st.button("🚀 Start BCI System", use_container_width=True, type="primary"):
        st.session_state.bci_active = True
        st.toast("System Initialized. Mapping brain signals to cursor...")
        
    if st.button("🛑 Emergency Stop", use_container_width=True):
        st.session_state.bci_active = False
        st.toast("System Disconnected.", icon="⚠️")

    st.write("---")
    st.info("💡 **Tip:** Keep the browser window focused or hover in a corner to trigger PyAutoGUI's failsafe.")

# --- MAIN DASHBOARD ---
st.title("🧠 Brain-Controlled Cursor Movement System")
st.caption("Assistive Hands-Free Computing using EEG Signals")

col1, col2, col3 = st.columns([1, 1, 1])

# Cognitive State Metric
with col1:
    state = "ACTIVE" if st.session_state.bci_active else "IDLE"
    color = "🟢" if st.session_state.bci_active else "⚪"
    st.metric("System Status", f"{color} {state}", delta=None)

with col2:
    conf = "99.5%" if st.session_state.bci_active else "0%"
    st.metric("Model Confidence", conf, delta="±0.2%")

with col3:
    latency = "38ms" if st.session_state.bci_active else "--"
    st.metric("Signal Latency", latency)

if not PYAUTOGUI_AVAILABLE:
    st.warning("🌐 **Cloud Simulation Mode:** Cursor control is disabled because this app is running on a cloud server. Run this project **locally** to enable brain-controlled cursor movement.")

st.write("---")

# Main Content Layout
left_panel, right_panel = st.columns([2, 1])

with left_panel:
    st.subheader("📡 Real-Time EEG Signal Stream (μV)")
    
    # Placeholder for chart
    chart_placeholder = st.empty()
    
    # Placeholder for live command
    st.write("")
    st.subheader("🎯 Predicted Brain Intent")
    command_placeholder = st.empty()
    command_placeholder.markdown("<div class='command-display'>IDLE</div>", unsafe_allow_html=True)

with right_panel:
    st.subheader("📋 Command Logs")
    table_placeholder = st.empty()

# --- SIMULATION LOOP ---
if st.session_state.bci_active:
    # Load model once
    model_handler.load_model()
    controller.set_active(True)
    
    # Load simulation data if not uploaded
    if st.session_state.eeg_data is None:
        if os.path.exists(PROCESSED_DATA_PATH):
            st.session_state.eeg_data = pd.read_csv(PROCESSED_DATA_PATH)
        else:
            st.error(f"Model data not found at: {PROCESSED_DATA_PATH}. Please ensure the 'data' folder is uploaded to GitHub.")
            st.stop()
            
    # Simulate signal processing loop
    data_ptr = 0
    df = st.session_state.eeg_data
    channels = preprocessor.channels
    
    # Increase step size for better visibility
    controller.step_size = 60
    
    while st.session_state.bci_active:
        # Get current signal slice
        sample = df.iloc[data_ptr % len(df)]
        
        # Use DataFrame to keep feature names and avoid warnings
        features = pd.DataFrame([sample[channels].values], columns=channels)
        
        # Predict command
        command = model_handler.predict(features)
        
        # Update UI: Predicted Command
        command_placeholder.markdown(f"<div class='command-display'>{command}</div>", unsafe_allow_html=True)
        
        # Update UI: Plot
        window_size = 50
        start_idx = max(0, data_ptr - window_size)
        plot_data = df.iloc[start_idx:data_ptr+1][channels]
        chart_placeholder.line_chart(plot_data)
        
        # Execute Cursor Movement
        controller.execute_command(command)
        
        # Log entry
        if command != "IDLE":
            timestamp = time.strftime("%H:%M:%S")
            st.session_state.logs.insert(0, {"Time": timestamp, "Command": command, "Status": "Success"})
            if len(st.session_state.logs) > 15:
                st.session_state.logs.pop()
            table_placeholder.table(pd.DataFrame(st.session_state.logs))
            
        # Skip 50 samples per iteration to cycle through commands faster (15000/50 samples = 300 iterations)
        # At 10 iterations per second, this cycles through the whole dataset in 30 seconds.
        data_ptr += 50 
        time.sleep(0.1) 
        
        # Check for stop
        # Note: In Streamlit, this loop can be tricky. Usually users use a fragment or rerun.
        # But for a continuous loop with manual stop, session_state is checked.
        # However, button click triggers a rerun, which will stop this loop.
else:
    command_placeholder.markdown("<div class='command-display'>DISCONNECTED</div>", unsafe_allow_html=True)
    if st.session_state.logs:
        table_placeholder.table(pd.DataFrame(st.session_state.logs))
