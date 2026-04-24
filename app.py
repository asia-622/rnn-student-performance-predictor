import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import tensorflow as tf
from model import StudentPerformanceLSTM
import os

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    /* [Previous CSS remains exactly the same] */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem 0; min-height: 100vh; }
    .stApp { background: transparent !important; }
    h1 { font-family: 'Poppins', sans-serif; font-weight: 700; color: white; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .metric-card { background: rgba(255, 255, 255, 0.95); padding: 1.5rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.2); }
    .prediction-card { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 2rem; border-radius: 20px; box-shadow: 0 20px 40px rgba(79, 172, 254, 0.4); text-align: center; font-size: 1.2rem; font-weight: 600; }
    .input-card { background: rgba(255, 255, 255, 0.95); padding: 1.5rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 10px; padding: 0.7rem 2rem; font-weight: 600; font-size: 1rem; transition: all 0.3s ease; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

# Check if model exists
if not os.path.exists('models/lstm_model.h5'):
    st.error("""
    🚫 **Model not found!**
    
    **Run these commands locally first:**
    ```bash
    pip install -r requirements.txt
    python train.py
    git add .
    git commit -m "Add trained model"
    git push
    ```
    """)
    st.stop()

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/lstm_model.h5')
    
    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    lstm_obj = StudentPerformanceLSTM()
    lstm_obj.model = model
    lstm_obj.scaler = scaler
    return lstm_obj

model_obj = load_model()

# Title & Main UI (same beautiful design as before)
st.markdown("""
# 🎓 Student Performance Trend Predictor
**Powered by LSTM Neural Network** 
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📊 Enter Weekly Performance")
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        week_cols = st.columns(4)
        week1 = week_cols[0].number_input("📚 Week 1", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
        week2 = week_cols[1].number_input("📈 Week 2", min_value=0.0, max_value=100.0, value=72.0, step=0.5)
        week3 = week_cols[2].number_input("📊 Week 3", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
        week4 = week_cols[3].number_input("🔥 Week 4", min_value=0.0, max_value=100.0, value=78.0, step=0.5)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("## 🎯 Prediction")
    
    if st.button("🔮 Predict Final Score", type="primary", use_container_width=True):
        input_data = np.array([week1, week2, week3, week4])
        
        with st.spinner("🔬 AI Analyzing trends..."):
            prediction = model_obj.predict(input_data)
        
        st.session_state.prediction = float(prediction)
        st.session_state.inputs = [week1, week2, week3, week4]
    
    if 'prediction' in st.session_state:
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🎯 Predicted Final Score</h2>
            <h1 style="font-size: 3rem; margin: 0;">{st.session_state.prediction:.1f}<span style="font-size: 1.5rem;">/100</span></h1>
            <p>📈 Expected Final Performance</p>
        </div>
        """, unsafe_allow_html=True)

# Chart (same as before)
if 'inputs' in st.session_state:
    st.markdown("## 📈 Performance Trend")
    fig = go.Figure()
    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4', '🎯 Final']
    scores = st.session_state.inputs + [st.session_state.prediction]
    
    fig.add_trace(go.Scatter(x=weeks, y=scores, mode='lines+markers', 
                            line=dict(color='#4facfe', width=4),
                            marker=dict(size=12, color='#00f2fe')))
    
    fig.update_layout(title="📊 Performance Evolution", height=400, showlegend=False,
                     template='plotly_white', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# Footer (same)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.8);'>
    <p>🎓 Production-Ready LSTM Predictor | Deployed on Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
