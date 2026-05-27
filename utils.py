import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

def apply_log_transform(df, columns):
    for col in columns:
        df[col] = np.log1p(df[col])
    return df

def create_sequences(data, n_lookback, n_forecast):
    X, y = [], []
    for i in range(len(data) - n_lookback - n_forecast + 1):
        X.append(data[i : (i + n_lookback)])
        y.append(data[(i + n_lookback) : (i + n_lookback + n_forecast)])
    return np.array(X), np.array(y)

def preprocess_data(csv_path, n_lookback=10, n_forecast=5):
    df = pd.read_csv(csv_path)
    features = ['Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', '%Deliverble']
    df = df[features].fillna(method='ffill').dropna()
    
    # Advanced Preprocessing: Log Transform on skewed columns
    skewed_cols = ['Volume', 'Turnover', 'Trades']
    df = apply_log_transform(df, skewed_cols)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    X, y = create_sequences(scaled_data, n_lookback, n_forecast)
    return X, y, scaler
