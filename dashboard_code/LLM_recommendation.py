import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score

# Function to fetch live stock data
def get_live_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1mo")
    return data

# Feature engineering
def engineer_features(df):
    df['moving_avg_30'] = df['Close'].rolling(window=30).mean()
    df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
    df['macd'], df['signal'], df['hist'] = talib.MACD(df['Close'], fastper=12, slowper=26, signalper=9)
    return df

# Define the models
models = {
    'LSTM': Sequential([
        LSTM(64, input_shape=(9, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ]),
    'GRU': Sequential([
        GRU(64, input_shape=(9, 1), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
}

# Compile and train the models
for name, model in models.items():
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    # Load and prepare the data
    df = get_live_data('AAPL')
    df = engineer_features(df)
    X = df[['Open', 'High', 'Low', 'Volume', 'moving_avg_30', 'rsi', 'macd', 'signal', 'hist']]
    y = df['Close']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model.fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1), y, epochs=50, batch_size=32, verbose=0)
    
    # Evaluate the model
    y_pred = model.predict(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1))
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    models[name]['mse'] = mse
    models[name]['r2'] = r2

# Streamlit app
st.title("Stock Price Prediction")

# Get the stock symbol from the user
stock_symbol = st.text_input("Enter the stock symbol:", "AAPL")

# Get the model selection from the user
model_name = st.selectbox("Select a deep learning model:", list(models.keys()))
selected_model = models[model_name]['model']

# Fetch the latest stock data for the given symbol
latest_data = get_live_data(stock_symbol)
latest_data = engineer_features(latest_data)
latest_data_scaled = scaler.transform(latest_data[['Open', 'High', 'Low', 'Volume', 'moving_avg_30', 'rsi', 'macd', 'signal', 'hist']].iloc[-1].to_frame().T)

# Make the prediction
prediction = selected_model.predict(latest_data_scaled.reshape(1, latest_data_scaled.shape[1], 1))

# Display the results
st.write(f"The predicted closing price for {stock_symbol} using the {model_name} model is: ${prediction[0][0]:.2f}")
st.write(f"Model MSE: {models[model_name]['mse']:.4f}, R-squared: {models[model_name]['r2']:.4f}")

# Analysis
st.subheader("Analysis")
st.write(f"You have selected the {model_name} deep learning model for stock price prediction.")
st.write("The LSTM and GRU models are well-suited for modeling time-series data like stock prices, as they can capture long-term dependencies and patterns.")
st.write("The model was trained on live stock data, including the key features like open, high, low, volume, moving average, RSI, MACD, and other technical indicators. By learning from this comprehensive set of features, the model can make more accurate predictions.")
st.write("The Streamlit app allows users to input a stock symbol and select the deep learning model they want to use. The predicted closing price and the model's accuracy metrics (MSE and R-squared) are displayed to the user.")
st.write("It's important to note that stock price prediction is a complex task, and these models may not be 100% accurate. Factors like unexpected news, global events, and other unpredictable market dynamics can impact stock prices in ways that are difficult for any model to capture. Users should always do their own research and use this as one of many inputs in their investment decision-making process.")
