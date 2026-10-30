import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def prepare_data(symbol='GOOG', start='2012-01-01', end='2022-12-21', seq_len=30):
    print(f"Downloading historical data for {symbol} ({start} to {end})...")
    raw = yf.download(symbol, start=start, end=end, progress=False)
    raw.dropna(inplace=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close'][symbol] if symbol in raw['Close'] else raw['Close'].iloc[:, 0]
        high = raw['High'][symbol] if symbol in raw['High'] else raw['High'].iloc[:, 0]
        low = raw['Low'][symbol] if symbol in raw['Low'] else raw['Low'].iloc[:, 0]
        volume = raw['Volume'][symbol] if symbol in raw['Volume'] else raw['Volume'].iloc[:, 0]
    else:
        close = raw['Close']
        high = raw['High']
        low = raw['Low']
        volume = raw['Volume']
        
    df = pd.DataFrame(index=raw.index)
    df['Close'] = close
    df['High'] = high
    df['Low'] = low
    df['Volume'] = volume
    
    # 1. Stationary relative returns
    df['Return'] = df['Close'].pct_change()
    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
    
    # 2. Moving Average Ratios
    df['MA10_Ratio'] = df['Close'] / df['Close'].rolling(10).mean() - 1.0
    df['MA20_Ratio'] = df['Close'] / df['Close'].rolling(20).mean() - 1.0
    df['MA50_Ratio'] = df['Close'] / df['Close'].rolling(50).mean() - 1.0
    
    # 3. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = (100.0 - (100.0 / (1.0 + rs))) / 100.0 - 0.5  # centered around 0
    
    # 4. MACD Ratio
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df['MACD_Ratio'] = macd / df['Close']
    
    # 5. Volume Change (clipped to avoid outliers)
    df['Vol_Change'] = df['Volume'].pct_change().clip(-1.0, 3.0)
    
    df.dropna(inplace=True)
    return df

def build_sequences(df, feature_cols, seq_len=30):
    features = df[feature_cols].values
    returns = df['Return'].values
    prices = df['Close'].values
    dates = df.index
    
    X, y_ret, p_prev, p_curr, dt = [], [], [], [], []
    for i in range(seq_len, len(df)):
        X.append(features[i-seq_len:i])
        y_ret.append(returns[i])
        p_prev.append(prices[i-1])
        p_curr.append(prices[i])
        dt.append(dates[i])
        
    return np.array(X), np.array(y_ret), np.array(p_prev), np.array(p_curr), dt

def directional_accuracy(y_true, y_pred, y_prev):
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    return float(np.mean(true_dir == pred_dir) * 100.0)

def main():
    symbol = 'GOOG'
    start_date = '2012-01-01'
    end_date = '2022-12-21'
    seq_len = 30
    
    df = prepare_data(symbol, start_date, end_date, seq_len)
    feature_cols = ['Return', 'HL_Spread', 'MA10_Ratio', 'MA20_Ratio', 'MA50_Ratio', 'RSI', 'MACD_Ratio', 'Vol_Change']
    
    X, y_ret, p_prev, p_curr, dates = build_sequences(df, feature_cols, seq_len)
    
    # 80/20 chronological train/test split
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_ret[:split], y_ret[split:]
    p_prev_train, p_prev_test = p_prev[:split], p_prev[split:]
    p_curr_train, p_curr_test = p_curr[:split], p_curr[split:]
    
    # Standardize features using training data ONLY (Zero Data Leakage)
    scaler = StandardScaler()
    N_tr, T, D = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, D)).reshape(N_tr, T, D)
    
    N_te = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(-1, D)).reshape(N_te, T, D)
    
    # Build Modernized Model
    model = Sequential([
        Input(shape=(T, D)),
        LSTM(48, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
        Dropout(0.2),
        LSTM(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )
    
    # Inference on Test Dataset
    pred_returns = model.predict(X_test_scaled, verbose=0).flatten()
    pred_prices = p_prev_test * (1.0 + pred_returns)
    actual_prices = p_curr_test
    
    # Model Metrics
    mae = float(mean_absolute_error(actual_prices, pred_prices))
    rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
    mape = float(np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100.0)
    r2 = float(r2_score(actual_prices, pred_prices))
    da = directional_accuracy(actual_prices, pred_prices, p_prev_test)
    
    # Baseline 1: Naive Persistence (predict today's price equals yesterday's price)
    mae_naive = float(mean_absolute_error(actual_prices, p_prev_test))
    rmse_naive = float(np.sqrt(mean_squared_error(actual_prices, p_prev_test)))
    mape_naive = float(np.mean(np.abs((actual_prices - p_prev_test) / actual_prices)) * 100.0)
    r2_naive = float(r2_score(actual_prices, p_prev_test))
    
    # Baseline 2: 20-day Simple Moving Average
    ma20_test = df['Close'].rolling(20).mean().iloc[-len(actual_prices):].values
    mae_ma = float(mean_absolute_error(actual_prices, ma20_test))
    rmse_ma = float(np.sqrt(mean_squared_error(actual_prices, ma20_test)))
    mape_ma = float(np.mean(np.abs((actual_prices - ma20_test) / actual_prices)) * 100.0)
    r2_ma = float(r2_score(actual_prices, ma20_test))
    
    metrics = {
        'symbol': symbol,
        'records_total': int(len(df)),
        'records_train': int(len(X_train)),
        'records_test': int(len(X_test)),
        'sequence_length': seq_len,
        'features': feature_cols,
        'lstm': {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2),
            'r2': round(r2, 4),
            'directional_accuracy': round(da, 2)
        },
        'naive_baseline': {
            'mae': round(mae_naive, 4),
            'rmse': round(rmse_naive, 4),
            'mape': round(mape_naive, 2),
            'r2': round(r2_naive, 4)
        },
        'ma20_baseline': {
            'mae': round(mae_ma, 4),
            'rmse': round(rmse_ma, 4),
            'mape': round(mape_ma, 2),
            'r2': round(r2_ma, 4)
        }
    }
    
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS REPORT")
    print("="*50)
    print(f"Test Samples: {len(actual_prices)}")
    print(f"LSTM Model      -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | R2: {r2:.4f} | DA: {da:.2f}%")
    print(f"Naive Baseline  -> MAE: {mae_naive:.4f} | RMSE: {rmse_naive:.4f} | MAPE: {mape_naive:.2f}% | R2: {r2_naive:.4f}")
    print(f"MA20 Baseline   -> MAE: {mae_ma:.4f} | RMSE: {rmse_ma:.4f} | MAPE: {mape_ma:.2f}% | R2: {r2_ma:.4f}")
    print("="*50 + "\n")
    
    # Save artifacts
    model.save('Stock Predictions Model.keras')
    joblib.dump(scaler, 'feature_scaler.joblib')
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Successfully saved:")
    print(" - Stock Predictions Model.keras")
    print(" - feature_scaler.joblib")
    print(" - metrics.json")

if __name__ == '__main__':
    main()
