"# Stock Market Price Predictor

An end-to-end machine learning web application for forecasting stock market closing prices using LSTM neural networks. This project combines historical stock data collection, time-series preprocessing, deep learning prediction, and an interactive Streamlit dashboard to visualize trends and model results.

## Project Overview

This project is designed to:

- Predict future stock closing prices using historical market data
- Analyze long-term price movement through technical indicators
- Show moving average trends for short-, mid-, and long-term decision support
- Provide an interactive web dashboard for stock analysis

The core model is built with TensorFlow/Keras and uses a Sequential LSTM architecture with a preprocessing pipeline based on Min-Max scaling and a 100-day lookback window.

## Objective

The goal is to forecast stock close prices by learning patterns from historical time-series data. The system uses the recent 100 days of scaled values as input and predicts price movement for the next time step, helping compare predicted and actual performance over a validation window.

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- yFinance
- Streamlit
- Matplotlib

## Repository Contents

- app.py — Streamlit web application for data loading, visualization, and prediction
- Stock_Market_Prediction_Model_Creation.ipynb — notebook used to explore and train the forecasting model
- README.md — project documentation

## How the App Works

The application follows this pipeline:

1. User enters a stock symbol such as GOOG, AAPL, or MSFT
2. Data is downloaded with yFinance for a configured date range
3. The close prices are normalized using MinMaxScaler
4. A 100-day sequence window is used to feed the LSTM model
5. The trained model predicts values on the test set
6. Results are plotted alongside the actual closing prices
7. Moving average charts are displayed for technical analysis

## Model Architecture

The model used in this project is based on an LSTM time-series forecasting architecture. In practice, the workflow includes:

- Historical stock close-price extraction
- Train/test split based on the last 20% of data
- Scaling to a range between 0 and 1
- Sequence generation using a 100-day window
- LSTM layers for sequential learning
- Dense output layer for price prediction

The implementation in the app assumes a pre-trained model file is available and loads it before running predictions.

## Features

### 1. Stock Data Retrieval
The app fetches real market data from Yahoo Finance using the yfinance library.

### 2. Historical Analysis
Users can inspect the dataset and analyze price movements from the specified date range.

### 3. Moving Averages
The dashboard visually compares:

- 50-day moving average
- 100-day moving average
- 200-day moving average

This helps interpret trend direction and momentum.

### 4. Price Prediction
Predicted price values are compared against actual close-price values for the test period. This gives a visual understanding of how well the model fits the stock history.

### 5. Interactive Dashboard
The app is built with Streamlit, making it easy to:

- Input a ticker symbol
- Inspect chart output
- Review historical data and technical indicators
- View prediction comparison

## Prerequisites

Before running this project, make sure you have Python installed on your system.

It is recommended to use a virtual environment for dependency management.

## Installation

Open a terminal in the project folder and install the required packages:

```bash
pip install numpy pandas yfinance streamlit matplotlib tensorflow scikit-learn
```

If you are using a Jupyter environment for model creation, also ensure notebook support is installed:

```bash
pip install notebook
```

## Running the Application

1. Clone or download the repository.
2. Open a terminal in the project directory.
3. Ensure the trained Keras model file is available.
4. Launch the app:

```bash
streamlit run app.py
```

After the app starts, a browser window should open with the stock predictor dashboard.

## Important Note About the Model File

The application currently loads a model using a hardcoded file path:

```python
model = load_model('C:\Python\Stock\Stock Predictions Model.keras')
```

This path is specific to the original machine used to create the project. If you run the app on a different computer, you should:

- place the trained model file in a local folder, and
- update the path in app.py to match the new location

Example:

```python
model = load_model('stock_predictions_model.keras')
```

or

```python
model = load_model(r'C:\path\to\your\model\stock_predictions_model.keras')
```

## Training the Model

The model is developed in the notebook file `Stock_Market_Prediction_Model_Creation.ipynb`.

The typical training workflow is:

1. Download historical price data
2. Select the close price column
3. Split data into training and testing sets
4. Normalize values using MinMaxScaler
5. Create sequences with a 100-day lag
6. Train the LSTM model
7. Save the trained model for use in the app

A typical saved model name could be:

```bash
stock_predictions_model.keras
```

## Data Source

This project uses Yahoo Finance via the yfinance package to obtain historical stock market data. The app targets a date range from 2012-01-01 to 2022-12-31 by default.

## Example Usage

In the app:

- Enter a stock symbol such as `GOOG`
- View the raw stock data table
- Examine moving average charts
- Review predicted vs. actual close prices

## Project Workflow Summary

The project combines:

- Data acquisition through yFinance
- Feature engineering with moving averages and scaling
- Deep learning with LSTM networks
- Model prediction and evaluation
- Visualization in a Streamlit interface

## Limitations and Disclaimer

This project is intended for educational and research purposes. Stock market prediction is inherently uncertain and influenced by many external factors, including news events, macroeconomic conditions, volatility, and market sentiment.

The model should not be treated as financial advice or a guaranteed forecasting system. Use it only as a learning tool or experimental prototype.

## Disclaimer

This project is for educational purposes only. Stock market prediction is highly volatile, and results may not be reliable for real-world financial decisions. Do not use this model as the sole basis for buying or selling stock.

## Future Improvements

Possible enhancements for this project include:

- Supporting multiple stock symbols in one dashboard
- Adding more technical indicators (RSI, MACD, Bollinger Bands)
- Training on more data and different timeframes
- Improving model accuracy with feature engineering
- Deploying the app online with cloud hosting
- Saving predictions and performance metrics in a report

## License

This project is provided as-is for learning and experimentation. Please check with the repository owner or relevant institution before using it in commercial or production contexts.

## Conclusion

The Stock Market Price Predictor is a practical demonstration of how time-series forecasting and deep learning can be applied to financial data. It shows the full workflow from data collection to visualization and prediction in a simple, interactive web app.

Whether you are learning LSTM networks, practicing time-series forecasting, or building a personal stock analysis dashboard, this project provides a strong foundation for experimentation and extension.
" 
