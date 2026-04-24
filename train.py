import pandas as pd
import numpy as np
from model import StudentPerformanceLSTM
import os
import pickle

def main():
    print("🚀 Training Student Performance LSTM Model...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load data
    df = pd.read_csv('data/student_performance.csv')
    
    # Prepare features (use first 4 weeks to predict week 5+)
    weeks = ['week1', 'week2', 'week3', 'week4']
    X = df[weeks].values
    y = df['final_score'].values
    
    # Create and train model
    model_obj = StudentPerformanceLSTM(timesteps=4)
    model_obj.scaler.fit(X.reshape(-1, 1))  # Fit scaler on all data
    
    print("📊 Training model...")
    history = model_obj.model.fit(
        X, y,
        epochs=50,  # Reduced for faster training
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    model_obj.model.save('models/lstm_model.h5')
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(model_obj.scaler, f)
    
    print("✅ Model and scaler saved!")
    
    # Test
    test_pred = model_obj.predict(X[0])
    print(f"Test - Actual: {y[0]:.1f}, Predicted: {test_pred:.1f}")

if __name__ == "__main__":
    main()
