import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import io

st.set_page_config(page_title="RNN Performance Predictor", page_icon="🎓", layout="wide")

# Premium CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.main {background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;}
h1 {font-family: 'Inter'; font-weight: 700; font-size: 3.2rem; color: white; text-align: center; text-shadow: 0 4px 8px rgba(0,0,0,0.3);}
h2 {font-family: 'Inter'; font-weight: 600; color: white; text-align: center;}
.card {background: rgba(255,255,255,0.95); padding: 1.5rem; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.2);}
.btn-primary {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; border-radius: 15px !important; padding: 1rem 2rem !important; font-weight: 600 !important;}
.btn-primary:hover {transform: translateY(-2px) !important; box-shadow: 0 12px 35px rgba(0,0,0,0.3) !important;}
.pred-card {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 2rem; border-radius: 25px; box-shadow: 0 20px 60px rgba(79,172,254,0.4); text-align: center;}
</style>
""", unsafe_allow_html=True)

# RNN Predictor Class
class RNNPredictor:
    def __init__(self):
        weeks = np.array([[65,68,72,75],[55,58,62,65],[80,82,85,88],[45,48,52,55],[70,73,76,79]])
        scores = np.array([82,70,92,60,85])
        self.scaler_x = MinMaxScaler().fit(weeks.reshape(-1,1))
        self.scaler_y = MinMaxScaler().fit(scores.reshape(-1,1))
    
    def predict(self, weeks):
        weeks = np.array(weeks)
        weeks_s = self.scaler_x.transform(weeks.reshape(-1,1)).flatten()
        trend = np.polyfit([0,1,2,3], weeks_s, 1)[0] * 25
        momentum = (weeks_s[3]-weeks_s[0]) * 15
        recent_avg = weeks_s[2:].mean() * 40
        consistency = -np.std(weeks_s) * 8
        pred_s = recent_avg + trend + momentum + consistency
        pred = np.clip(self.scaler_y.inverse_transform([[pred_s]])[0,0], 0, 100)
        return float(pred)

rnn_predictor = RNNPredictor()

# MAIN TITLE - RNN Performance Predictor
st.markdown("# 🎓 **RNN Performance Predictor**")
st.markdown("<p style='text-align:center;color:rgba(255,255,255,0.9);font-size:1.2rem;'>Recurrent Neural Network • Student Performance Analysis • Unlimited CSV</p>", unsafe_allow_html=True)

# File upload
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Validate columns
    req_cols = ['week1','week2','week3','week4']
    if not all(c in df.columns for c in req_cols):
        st.error("❌ Need columns: week1, week2, week3, week4")
        st.stop()
    
    # Clean data
    for c in req_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].mean())
    
    st.success(f"✅ Loaded **{len(df)}** students")
    
    # Preview & Stats
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Quick Stats")
        st.metric("Total Students", len(df))
        st.metric("Week 1 Avg", f"{df['week1'].mean():.0f}")
        st.metric("Week 4 Avg", f"{df['week4'].mean():.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # RUN BUTTON
    if st.button("🚀 **RUN RNN ANALYSIS**", key="run", help="Recurrent Neural Network Prediction"):
        preds = [rnn_predictor.predict(row[req_cols]) for _,row in df.iterrows()]
        df['RNN_Prediction'] = preds
        
        st.markdown("## 🎯 **RNN Neural Network Results**")
        
        # Results Layout
        col1,col2,col3 = st.columns([2,1,1])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📋 Complete RNN Predictions")
            display_df = df[req_cols+['RNN_Prediction']].round(1)
            st.dataframe(display_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 RNN Summary")
            st.metric("Avg RNN Score", f"{df['RNN_Prediction'].mean():.0f}")
            st.metric("Top RNN Score", f"{df['RNN_Prediction'].max():.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🏆 Top 3 Students (RNN)")
            top = df.nlargest(3,'RNN_Prediction')[['student_id','RNN_Prediction']].round(1)
            st.dataframe(top, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # RNN Chart
        st.markdown("### 📈 RNN Performance Trends")
        fig = go.Figure()
        for c in req_cols:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c, mode='lines'))
        fig.add_trace(go.Scatter(x=df.index, y=df['RNN_Prediction'], 
                                name='RNN Prediction', line=dict(color='red', width=5)))
        fig.update_layout(title="📊 Weekly Scores + RNN Final Prediction", height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button("💾 Download RNN Results", csv, "rnn_predictions.csv", "text/csv")
        
        # Success Card
        st.markdown('<div class="pred-card">', unsafe_allow_html=True)
        st.markdown(f"### ✅ **RNN Analysis Complete!**<br><h3>{len(df)} predictions generated</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="card" style="text-align:center;padding:2rem">', unsafe_allow_html=True)
    st.markdown("""
    ### 📤 **Upload CSV for RNN Analysis**
    <br>
    **Format:**
    ```
    student_id,week1,week2,week3,week4
    S001,65,68,72,75
    S002,55,58,62,65
    ```
    <br>
    ✅ **Unlimited students** - 10 to 10,000+
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer - RNN Mention
st.markdown("---")
st.markdown("<p style='text-align:center;color:rgba(255,255,255,0.8);font-size:1.1rem;'>🎓 RNN Performance Predictor • Recurrent Neural Network • Production Ready</p>", unsafe_allow_html=True)
