import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

class StudentPerformanceLSTM:
    def __init__(self, timesteps=4, features=1):
        self.timesteps = timesteps
        self.features = features
        self.scaler = MinMaxScaler()
        self.model = self.build_model()
    
    def build_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.timesteps, self.features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        return model
    
    def prepare_data(self, data):
        """Prepare time series data for LSTM"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.timesteps, len(scaled_data)):
            X.append(scaled_data[i-self.timesteps:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaled_data
    
    def predict(self, input_sequence):
        """Make prediction on new data"""
        # Scale input
        input_scaled = self.scaler.transform(np.array(input_sequence).reshape(-1, 1))
        input_scaled = input_scaled.reshape(1, self.timesteps, 1)
        
        # Predict
        pred_scaled = self.model.predict(input_scaled, verbose=0)
        
        # Inverse transform
        pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
        return float(pred)
