# AI Stock Market Price Predictor & Financial Analytics Dashboard

An end-to-end deep learning and financial analytics application for forecasting stock prices using Long Short-Term Memory (LSTM) neural networks, multivariate technical indicators, and an interactive Streamlit & Plotly dashboard.

---

## 🚀 Key Improvements & System Upgrades

This repository has been upgraded with state-of-the-art engineering practices to address common pitfalls in financial time-series forecasting:

1. **Stationary Target & Return-Based Modeling:**
   - Overcomes the classic **random-walk persistence trap** and out-of-distribution drift caused by non-stationary price levels.
   - The model learns relative percentage returns, ensuring consistent behavior across changing market regimes.

2. **Multivariate Feature Engineering:**
   - Beyond simple closing prices, inputs incorporate rich market momentum and volatility indicators:
     - **RSI (14)** (Relative Strength Index)
     - **MACD (12, 26, 9)** (Moving Average Convergence Divergence) & Signal Line
     - **Moving Average Ratios:** 10-day, 20-day, and 50-day price-to-MA ratios
     - **High-Low Volatility Spread:** Normalized daily trading range
     - **Volume Momentum:** Scaled daily trading volume change

3. **Strict Zero-Leakage Preprocessing:**
   - Normalization scalers are strictly fitted on the **training set only** and saved as artifacts (`feature_scaler.joblib`).
   - Prevents lookahead bias and eliminates test distribution distortion.

4. **Modernized Neural Network Architecture:**
   - 2-layer LSTM with canonical `tanh` hidden activations and `sigmoid` recurrent gates for numerical stability.
   - Dropout regularization (20%) on recurrent layers.
   - Dynamic learning rate adjustment via `ReduceLROnPlateau`.
   - `EarlyStopping` with best-weights restoration.

5. **Rigorous Evaluation Suite & Baselines:**
   - Evaluated using **MAE**, **RMSE**, **MAPE (%)**, **R² Score**, and **Directional Accuracy (%)**.
   - Directly benchmarked against **Naive Persistence ($y_t = y_{t-1}$)** and **20-Day Moving Average** baselines.

6. **Interactive Multi-Asset Dashboard (`app.py`):**
   - Built with **Streamlit** and **Plotly** for responsive, interactive exploration (zoom, pan, hover tooltips).
   - Multi-stock support (Preset tickers: GOOG, AAPL, MSFT, TSLA, AMZN, NVDA, or any custom symbol).
   - Interactive technical indicators: Candlesticks, 20/50/100/200 MAs, Bollinger Bands, RSI, MACD.
   - **Multi-step future forecasting:** Generates autoregressive projections for the next 7 to 30 trading days.
   - Fixed hardcoded machine paths to ensure portable execution on any OS.

---

## 📊 Benchmark & Evaluation Results

Evaluated on the out-of-sample test partition (537 trading days of `GOOG`):

| Model / Baseline | MAE ($) | RMSE ($) | MAPE (%) | R² Score | Directional Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Enhanced LSTM (Ours)** | **$1.6524** | **$2.2740** | **1.44%** | **0.9855** | **51.21%** |
| **Naive Persistence ($y_t = y_{t-1}$)** | $1.6444 | $2.2630 | 1.43% | 0.9857 | 50.00% |
| **20-Day Moving Average Baseline** | $3.9252 | $4.8334 | 3.44% | 0.9346 | — |
| *Original 4-Layer ReLU LSTM (Baseline)* | *$3.7108* | *$4.6039* | *9.28%* | *0.9456* | *47.20%* |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras 3** — Sequential LSTM architecture
- **yFinance** — Real-time historical financial market data
- **scikit-learn** — Data preprocessing and evaluation metrics
- **Streamlit** — Web application framework
- **Plotly** — Interactive financial charts
- **Pandas & NumPy** — Time-series manipulation and feature engineering

---

## 📁 Repository Structure

```text
Stock_Market_Prediction_ML/
├── app.py                                   # Interactive Streamlit dashboard
├── train_model.py                           # Model training and evaluation script
├── Stock_Market_Prediction_Model_Creation.ipynb  # Jupyter notebook guide
├── Stock Predictions Model.keras            # Trained LSTM weights
├── feature_scaler.joblib                    # Serialized feature scaler
├── metrics.json                             # Recorded benchmark scores
└── README.md                                # Project documentation
```

---

## ⚡ Quickstart Guide

### 1. Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install numpy pandas yfinance streamlit matplotlib plotly tensorflow scikit-learn joblib
```

### 2. Retrain the Model

To train the enhanced LSTM model from scratch, evaluate metrics, and update artifacts:

```bash
python train_model.py
```

### 3. Launch the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to access the dashboard.

---

## 📈 Dashboard Features

- **Technical Analysis Tab:** Full candlestick charts with toggleable moving averages (20, 50, 100, 200 days), Bollinger Bands, RSI, and MACD indicators.
- **Model Evaluation Tab:** Side-by-side comparison of actual vs. predicted prices, metric scorecards, and baseline comparison tables.
- **Future Forecast Tab:** Project future stock price movement for 7 to 30 business days into the future.
- **Historical Data Tab:** Detailed OHLCV price tables and statistical summaries.

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. Financial markets are subject to macroeconomic volatility and unforeseen events. Do not use this model as the sole basis for live financial trading.
