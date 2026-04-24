import pandas as pd
import numpy as np
from model import StudentPerformanceLSTM
import os

def main():
    print("🚀 Training Student Performance LSTM Model...")
    
    # Load data
    df = pd.read_csv('data/student_performance.csv')
    
    # Prepare features and target
    weeks = ['week1', 'week2', 'week3', 'week4', 'week5', 'week6', 'week7', 'week8']
    X = df[weeks].values
    y = df['final_score'].values
    
    # Create model
    model = StudentPerformanceLSTM(timesteps=4)
    
    print("📊 Training model...")
    history = model.model.fit(
        X, y,
        epochs=100,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.model.save('models/lstm_model.h5')
    print("✅ Model saved as 'models/lstm_model.h5'")
    
    # Test prediction
    test_input = X[0][:4]
    prediction = model.predict(test_input)
    print(f"Test Prediction: {prediction:.2f} (Actual: {y[0]:.2f})")

if __name__ == "__main__":
    main()
