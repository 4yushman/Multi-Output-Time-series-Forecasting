# 🚀 Multivariate Time-Series Stock Forecasting using LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced deep learning project aimed at predicting multi-output stock market indicators for **NIFTY-50 (BAJFINANCE)** using Long Short-Term Memory (LSTM) networks. This project implements a many-to-many sequence architecture to forecast 11 market features for a 5-day horizon.

---

## 🌟 Key Highlights
- **Multi-Output Forecasting:** Predicts 11 features (Open, High, Low, Volume, etc.) simultaneously for the next 5 trading days.
- **Advanced Preprocessing:** Implemented **Log-Normal Transformation** to stabilize variance in highly skewed financial data.
- **GPU Accelerated:** Optimized for **NVIDIA Tesla T4 (CUDA)**, ensuring high-speed tensor computations during the 150-epoch training cycle.
- **Robust Pipeline:** Uses a Sliding Window approach (10-day lookback) with strictly isolated Training/Validation splits to prevent data leakage.

---

## 📊 Visualizations

### 1. Training vs Validation Loss
*The model demonstrates smooth convergence, highlighting the effectiveness of the Learning Rate Scheduler and Dropout layers.*
![Loss Graph](images/loss_plot.png) <!-- Update this path after uploading your screenshot -->

### 2. Actual vs Predicted Price (Close)
*A comparison showcasing the model's ability to capture complex market trends and volatility.*
![Prediction Graph](images/prediction_plot.png) <!-- Update this path after uploading your screenshot -->

---

## 🛠️ Tech Stack
- **Core:** Python, PyTorch (Deep Learning Framework)
- **Data:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Kaggle GPU (Tesla T4)

---

## 🧠 Model Architecture
The core model is a multi-layered **LSTM** designed to handle sequential dependencies:
- **Input Dim:** 11 features
- **Hidden Layers:** 2 LSTM layers with 64/32 units
- **Regularization:** Dropout (0.2) to prevent overfitting
- **Output Dim:** 55 (11 features × 5 days)

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch pandas numpy matplotlib scikit-learn
