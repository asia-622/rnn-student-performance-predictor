import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import StudentPerformanceLSTM
import io

# Page config
st.set_page_config(
    page_title="RNN Student Performance Evaluator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
/* Main background */
.main .block-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Title styling */
h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 3rem;
    color: white;
    text-align: center;
    text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
}

h2 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: white;
    text-align: center;
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.95);
    padding: 1.5rem;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    text-align: center;
}

.prediction-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 2rem;
    border-radius: 25px;
    box-shadow: 0 20px 60px rgba(79,172,254,0.4);
    text-align: center;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 1rem 2rem;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(0,0,0,0.3);
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
# 🎓 **RNN Student Performance Evaluator**
## Powered by **LSTM Neural Network** • File Upload • Unlimited Data
""")

# Initialize model
@st.cache_resource
def load_lstm_model():
    model = StudentPerformanceLSTM()
    model.load_model()
    return model

lstm_model = load_lstm_model()

# Sidebar
st.sidebar.markdown("## 📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose CSV file", type="csv", 
    help="Upload student performance data (week1, week2, week3, week4, final_score)"
)

# Main content
if uploaded_file is not None:
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{len(df)}** student records!")
        
        # Check required columns
        required_cols = ['week1', 'week2', 'week3', 'week4']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.stop()
        
        # Data preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Quick Stats")
            st.metric("Total Students", len(df))
            st.metric("Avg Week 1", f"{df['week1'].mean():.1f}")
            st.metric("Avg Week 4", f"{df['week4'].mean():.1f}")
        
        # Prediction button
        if st.button("🚀 **RUN LSTM ANALYSIS**", type="primary", use_container_width=True):
            with st.spinner("🔬 LSTM model analyzing performance trends..."):
                # Make predictions
                predictions = []
                for idx, row in df.iterrows():
                    weeks = [row['week1'], row['week2'], row['week3'], row['week4']]
                    pred = lstm_model.predict(weeks)
                    predictions.append(pred)
                
                df['predicted_final_score'] = predictions
            
            # Results
            st.markdown("---")
            st.markdown("## 🎯 **LSTM Predictions**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📋 Complete Results")
                st.dataframe(
                    df[['student_id', 'week1', 'week2', 'week3', 'week4', 'predicted_final_score']],
                    use_container_width=True
                )
            
            with col2:
                st.markdown("### 📊 Summary Metrics")
                st.metric("Total Students", len(df))
                st.metric("Avg Predicted Score", f"{df['predicted_final_score'].mean():.1f}")
                st.metric("Highest Prediction", f"{df['predicted_final_score'].max():.1f}")
            
            with col3:
                st.markdown("### 📈 Top Performers")
                top_students = df.nlargest(5, 'predicted_final_score')[['student_id', 'predicted_final_score']]
                st.dataframe(top_students, use_container_width=True)
            
            # Trend chart
            st.markdown("### 📈 Performance Trends")
            fig = px.line(df, x='student_id', y=['week1', 'week2', 'week3', 'week4', 'predicted_final_score'],
                         title="Weekly Performance + LSTM Prediction",
                         markers=True)
            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "💾 Download Results CSV",
                csv_buffer.getvalue(),
                "lstm_predictions.csv",
                "text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("💡 Make sure your CSV has columns: student_id, week1, week2, week3, week4, final_score")

else:
    st.info("👆 **Please upload a CSV file** with student performance data")
    st.markdown("""
    ### 📋 **Expected CSV Format:**
    ```
    student_id,week1,week2,week3,week4,final_score
    S001,65,68,72,75,82
    S002,55,58,62,65,70
    ```
    **Works with ANY number of students!** (100, 1000, 10,000+)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.8); padding: 2rem;'>
    <h3>🎓 RNN-LSTM Student Performance Evaluator</h3>
    <p>Production-Ready • Unlimited Data • Neural Network Powered</p>
</div>
""", unsafe_allow_html=True)
