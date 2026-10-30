import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
from tensorflow.keras.models import load_model
import yfinance as yf

# Set Page Config
st.set_page_config(
    page_title="AI Stock Market Predictor & Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_trained_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'Stock Predictions Model.keras')
    scaler_path = os.path.join(base_dir, 'feature_scaler.joblib')
    metrics_path = os.path.join(base_dir, 'metrics.json')
    
    model = None
    scaler = None
    metrics = None
    
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
        except Exception as e:
            st.warning(f"Could not load Keras model: {e}")
            
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Could not load scaler: {e}")
            
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
            
    return model, scaler, metrics

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
            
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close'][ticker] if ticker in data['Close'] else data['Close'].iloc[:, 0]
            high = data['High'][ticker] if ticker in data['High'] else data['High'].iloc[:, 0]
            low = data['Low'][ticker] if ticker in data['Low'] else data['Low'].iloc[:, 0]
            open_p = data['Open'][ticker] if ticker in data['Open'] else data['Open'].iloc[:, 0]
            volume = data['Volume'][ticker] if ticker in data['Volume'] else data['Volume'].iloc[:, 0]
        else:
            close = data['Close']
            high = data['High']
            low = data['Low']
            open_p = data['Open']
            volume = data['Volume']
            
        clean_df = pd.DataFrame(index=data.index)
        clean_df['Open'] = open_p
        clean_df['High'] = high
        clean_df['Low'] = low
        clean_df['Close'] = close
        clean_df['Volume'] = volume
        clean_df.dropna(inplace=True)
        return clean_df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def compute_indicators(df):
    data = df.copy()
    # Moving Averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA100'] = data['Close'].rolling(window=100).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # RSI (14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    # MACD (12, 26, 9)
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # Bollinger Bands (20-day, 2 std)
    std20 = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['MA20'] + (std20 * 2)
    data['BB_Lower'] = data['MA20'] - (std20 * 2)
    
    # Model Input Ratios
    data['Return'] = data['Close'].pct_change()
    data['HL_Spread'] = (data['High'] - data['Low']) / data['Close']
    data['MA10_Ratio'] = data['Close'] / data['Close'].rolling(10).mean() - 1.0
    data['MA20_Ratio'] = data['Close'] / data['MA20'] - 1.0
    data['MA50_Ratio'] = data['Close'] / data['MA50'] - 1.0
    data['RSI_Scaled'] = data['RSI'] / 100.0 - 0.5
    data['MACD_Ratio'] = data['MACD'] / data['Close']
    data['Vol_Change'] = data['Volume'].pct_change().clip(-1.0, 3.0)
    
    return data

def forecast_future(model, scaler, recent_df, feature_cols, forecast_days=14, seq_len=30):
    curr_df = recent_df.copy()
    future_predictions = []
    last_date = curr_df.index[-1]
    
    for day in range(forecast_days):
        # Extract last seq_len features
        seq_features = curr_df[feature_cols].iloc[-seq_len:].values
        seq_scaled = scaler.transform(seq_features).reshape(1, seq_len, len(feature_cols))
        
        pred_ret = float(model.predict(seq_scaled, verbose=0)[0, 0])
        last_close = float(curr_df['Close'].iloc[-1])
        new_close = last_close * (1.0 + pred_ret)
        
        # Advance date to next business day
        last_date = last_date + datetime.timedelta(days=1)
        while last_date.weekday() >= 5:  # skip weekends
            last_date = last_date + datetime.timedelta(days=1)
            
        future_predictions.append({
            'Date': last_date,
            'Predicted_Close': new_close,
            'Predicted_Return_%': pred_ret * 100
        })
        
        # Append estimated new row to historical buffer
        new_row = pd.DataFrame([{
            'Open': new_close,
            'High': new_close * 1.005,
            'Low': new_close * 0.995,
            'Close': new_close,
            'Volume': curr_df['Volume'].iloc[-1]
        }], index=[last_date])
        
        # Concatenate and dynamically recalculate all technical indicators for the extended series
        extended_df = pd.concat([curr_df[['Open', 'High', 'Low', 'Close', 'Volume']], new_row])
        curr_df = compute_indicators(extended_df)
        
    return pd.DataFrame(future_predictions).set_index('Date')

# Sidebar Navigation & Inputs
st.sidebar.title("Configuration")

ticker_mode = st.sidebar.radio("Select Stock Mode:", ["Preset Popular", "Custom Ticker"])
if ticker_mode == "Preset Popular":
    symbol = st.sidebar.selectbox("Choose Stock:", ["GOOG", "AAPL", "MSFT", "AMZN", "TSLA", "NVDA"], index=0)
else:
    symbol = st.sidebar.text_input("Enter Ticker Symbol:", "GOOG").upper().strip()

col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = col_d2.date_input("End Date", datetime.date(2023, 1, 1))

forecast_days = st.sidebar.slider("Future Forecast Horizon (Days)", min_value=7, max_value=30, value=14, step=7)

st.sidebar.markdown("---")
st.sidebar.subheader("Chart Overlays")
show_ma20 = st.sidebar.checkbox("20-Day Moving Average", value=True)
show_ma50 = st.sidebar.checkbox("50-Day Moving Average", value=True)
show_ma100 = st.sidebar.checkbox("100-Day Moving Average", value=False)
show_ma200 = st.sidebar.checkbox("200-Day Moving Average", value=False)
show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=True)

# Main Title & Data Loading
st.title("📈 Advanced Stock Market Prediction & Analytics")
st.caption(f"Interactive technical indicators, LSTM deep learning forecasting, and evaluation benchmark for **{symbol}**.")

model, scaler, saved_metrics = load_trained_model()

with st.spinner(f"Loading market data for {symbol}..."):
    raw_df = fetch_stock_data(symbol, start_date, end_date)

if raw_df is None or len(raw_df) < 60:
    st.error(f"Insufficient historical data found for {symbol}. Please select an earlier start date or a valid ticker.")
    st.stop()

df = compute_indicators(raw_df)
feature_cols = ['Return', 'HL_Spread', 'MA10_Ratio', 'MA20_Ratio', 'MA50_Ratio', 'RSI_Scaled', 'MACD_Ratio', 'Vol_Change']

# Tabs Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Technical Analysis", 
    "🤖 Model Evaluation & Test Predictions", 
    "🔮 Multi-Day Future Forecast", 
    "📋 Historical Data"
])

# ----------------- TAB 1: TECHNICAL ANALYSIS -----------------
with tab1:
    st.subheader(f"{symbol} Price Action & Indicators")
    
    # Price and Overlays Chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price (OHLC)", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ))
    
    if show_ma20:
        fig_price.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#ff9800', width=1.5), name="MA20"))
    if show_ma50:
        fig_price.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='#2196f3', width=1.5), name="MA50"))
    if show_ma100:
        fig_price.add_trace(go.Scatter(x=df.index, y=df['MA100'], line=dict(color='#9c27b0', width=1.5), name="MA100"))
    if show_ma200:
        fig_price.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='#e91e63', width=1.5), name="MA200"))
    if show_bb:
        fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name="BB Upper"))
        fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)', name="BB Lower"))

    fig_price.update_layout(
        title=f"{symbol} Candlestick Chart with Technical Indicators",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Oscillators: RSI and MACD
    col_rsi, col_macd = st.columns(2)
    
    with col_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ab47bc', width=1.8), name="RSI (14)"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(title="Relative Strength Index (RSI 14)", yaxis=dict(range=[0, 100]), height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_rsi, use_container_width=True)
        
    with col_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#1976d2', width=1.5), name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#ff7043', width=1.5), name="Signal"))
        fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=np.where(df['MACD_Hist']>=0, '#26a69a', '#ef5350'), name="Histogram"))
        fig_macd.update_layout(title="MACD (12, 26, 9)", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_macd, use_container_width=True)

# ----------------- TAB 2: MODEL EVALUATION -----------------
with tab2:
    st.subheader("Model Evaluation & Historical Benchmark")
    
    if model is None or scaler is None:
        st.warning("Pre-trained model (`Stock Predictions Model.keras`) or scaler not found in workspace. Run `python train_model.py` to train.")
    else:
        # Prepare test sequences
        clean_features = df[feature_cols].dropna()
        valid_idx = clean_features.index
        prices = df.loc[valid_idx, 'Close'].values
        feat_vals = clean_features.values
        
        seq_len = 30
        if len(feat_vals) > seq_len + 30:
            split_idx = int(len(feat_vals) * 0.80)
            X_test, y_actual, p_prev_test, test_dates = [], [], [], []
            
            for i in range(max(split_idx, seq_len), len(feat_vals)):
                X_test.append(feat_vals[i-seq_len:i])
                y_actual.append(prices[i])
                p_prev_test.append(prices[i-1])
                test_dates.append(valid_idx[i])
                
            X_test = np.array(X_test)
            y_actual = np.array(y_actual)
            p_prev_test = np.array(p_prev_test)
            
            # Predict
            N_t, T_t, D_t = X_test.shape
            X_test_scaled = scaler.transform(X_test.reshape(-1, D_t)).reshape(N_t, T_t, D_t)
            pred_returns = model.predict(X_test_scaled, verbose=0).flatten()
            y_pred = p_prev_test * (1.0 + pred_returns)
            
            # Metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100.0
            r2 = r2_score(y_actual, y_pred)
            da = np.mean(np.sign(y_actual - p_prev_test) == np.sign(y_pred - p_prev_test)) * 100.0
            
            # Naive baseline
            mae_naive = mean_absolute_error(y_actual, p_prev_test)
            rmse_naive = np.sqrt(mean_squared_error(y_actual, p_prev_test))
            mape_naive = np.mean(np.abs((y_actual - p_prev_test) / y_actual)) * 100.0
            r2_naive = r2_score(y_actual, p_prev_test)
            
            # KPI Cards
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MAE", f"${mae:.2f}", delta=f"{mae - mae_naive:.2f} vs Naive", delta_color="inverse")
            c2.metric("RMSE", f"${rmse:.2f}", delta=f"{rmse - rmse_naive:.2f} vs Naive", delta_color="inverse")
            c3.metric("MAPE", f"{mape:.2f}%", delta=f"{mape - mape_naive:.2f}% vs Naive", delta_color="inverse")
            c4.metric("R² Score", f"{r2:.4f}")
            c5.metric("Directional Acc.", f"{da:.1f}%")
            
            # Actual vs Predicted Plot
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(
                x=test_dates, y=y_actual, mode='lines', line=dict(color='#2e7d32', width=2), name="Actual Price"
            ))
            fig_eval.add_trace(go.Scatter(
                x=test_dates, y=y_pred, mode='lines', line=dict(color='#d32f2f', width=1.8, dash='dash'), name="LSTM Predicted Price"
            ))
            fig_eval.update_layout(
                title=f"Historical Test Prediction vs. Actual Price for {symbol}",
                yaxis_title="Price ($)",
                height=450,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_eval, use_container_width=True)
            
            # Comparison Summary Table
            st.markdown("#### Baseline Comparison Table")
            comp_df = pd.DataFrame({
                "Model / Baseline": ["Enhanced LSTM", "Naive Persistence (y_t = y_{t-1})", "20-Day Moving Average"],
                "MAE ($)": [f"{mae:.4f}", f"{mae_naive:.4f}", f"{mean_absolute_error(y_actual, df.loc[test_dates, 'MA20']):.4f}"],
                "RMSE ($)": [f"{rmse:.4f}", f"{rmse_naive:.4f}", f"{np.sqrt(mean_squared_error(y_actual, df.loc[test_dates, 'MA20'])):.4f}"],
                "MAPE (%)": [f"{mape:.2f}%", f"{mape_naive:.2f}%", f"{np.mean(np.abs((y_actual - df.loc[test_dates, 'MA20'])/y_actual))*100:.2f}%"],
                "R² Score": [f"{r2:.4f}", f"{r2_naive:.4f}", f"{r2_score(y_actual, df.loc[test_dates, 'MA20']):.4f}"]
            })
            st.table(comp_df)
        else:
            st.info("Dataset too small for 80/20 train/test evaluation split. Select an earlier start date.")

# ----------------- TAB 3: FUTURE FORECAST -----------------
with tab3:
    st.subheader(f"🔮 {forecast_days}-Day Future Forecast for {symbol}")
    st.info("Multi-step autoregressive projection using the trained LSTM model.")
    
    if model is None or scaler is None:
        st.warning("Pre-trained model or scaler not loaded.")
    else:
        recent_clean = df[feature_cols + ['Close', 'Volume']].dropna()
        if len(recent_clean) >= 30:
            forecast_df = forecast_future(model, scaler, recent_clean, feature_cols, forecast_days=forecast_days, seq_len=30)
            
            # Projection Chart
            recent_hist = df[['Close']].iloc[-60:]
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=recent_hist.index, y=recent_hist['Close'], mode='lines', line=dict(color='#1f77b4', width=2), name="Recent Actual Price"
            ))
            fig_fc.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df['Predicted_Close'], mode='lines+markers', line=dict(color='#ff7f0e', width=2.5, dash='dash'), name=f"{forecast_days}-Day AI Forecast"
            ))
            fig_fc.update_layout(
                title=f"Future Price Trajectory for {symbol}",
                yaxis_title="Stock Price ($)",
                height=450,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            
            # Forecast Table
            st.markdown("#### Projected Daily Forecast")
            forecast_display = forecast_df.copy()
            forecast_display['Predicted_Close'] = forecast_display['Predicted_Close'].map('${:,.2f}'.format)
            forecast_display['Predicted_Return_%'] = forecast_display['Predicted_Return_%'].map('{:+.2f}%'.format)
            st.dataframe(forecast_display, use_container_width=True)
        else:
            st.warning("Insufficient data points to generate future forecasts.")

# ----------------- TAB 4: HISTORICAL DATA -----------------
with tab4:
    st.subheader(f"Historical Records ({symbol})")
    st.dataframe(raw_df.tail(100), use_container_width=True)
    st.markdown("#### Summary Statistics")
    st.dataframe(raw_df.describe(), use_container_width=True)
