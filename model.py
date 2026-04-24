import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os

class StudentPerformanceLSTM:
    def __init__(self, timesteps=4):
        self.timesteps = timesteps
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model_path = 'models/lstm_model.h5'
        
    def build_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.timesteps, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def prepare_training_data(self, df):
        """Prepare data for LSTM training"""
        # Use first 4 weeks as input, final_score as target
        X = df[['week1', 'week2', 'week3', 'week4']].values
        y = df['final_score'].values.reshape(-1, 1)
        
        # Reshape X for LSTM [samples, timesteps, features]
        X_reshaped = X.reshape(X.shape[0], self.timesteps, 1)
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X_reshaped.reshape(-1, 1)).reshape(X_reshaped.shape)
        y_scaled = self.scaler_y.fit_transform(y)
        
        return X_scaled, y_scaled
    
    def train(self, df, epochs=50):
        """Train the LSTM model"""
        os.makedirs('models', exist_ok=True)
        
        X, y = self.prepare_training_data(df)
        
        self.model = self.build_model()
        history = self.model.fit(X, y, epochs=epochs, batch_size=8, 
                               validation_split=0.2, verbose=1)
        
        self.model.save(self.model_path)
        print(f"✅ Model saved to {self.model_path}")
        return history
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            return True
        return False
    
    def predict(self, weeks_data):
        """Predict final score for new data"""
        if not hasattr(self, 'model'):
            self.load_model()
        
        # Prepare input
        X = np.array(weeks_data).reshape(1, self.timesteps, 1)
        X_scaled = self.scaler_X.transform(X.reshape(-1, 1)).reshape(X.shape)
        
        # Predict
        pred_scaled = self.model.predict(X_scaled, verbose=0)
        pred = self.scaler_y.inverse_transform(pred_scaled)[0, 0]
        
        return float(pred)
