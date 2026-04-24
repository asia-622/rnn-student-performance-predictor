import pandas as pd
from model import StudentPerformanceLSTM

def main():
    print("🚀 Training LSTM Student Performance Model...")
    
    # Load sample data
    df = pd.read_csv('data/sample_dataset.csv')
    print(f"📊 Loaded {len(df)} student records")
    
    # Train model
    lstm_model = StudentPerformanceLSTM()
    history = lstm_model.train(df, epochs=50)
    
    # Test prediction
    test_weeks = [[65, 68, 72, 75]]
    prediction = lstm_model.predict(test_weeks[0])
    print(f"🧪 Test Prediction: {prediction:.2f}")
    
    print("🎉 Training complete! Model ready for deployment.")

if __name__ == "__main__":
    main()
