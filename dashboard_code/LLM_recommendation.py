import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to load data
def load_data(ticker):
    data = yf.download(ticker, period='1y', interval = '1d')
    st.write(data)
    # Reset the index to remove the MultiIndex
    data.reset_index(inplace=True)
    # Assuming 'data' has a MultiIndex, drop the second level of the MultiIndex
    data.columns = data.columns.droplevel(1)
    return data

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create training and testing datasets
def create_datasets(scaled_data):
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_test, y_test = [], []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        y_test.append(test_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test

# Function to build LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to make predictions and evaluate the model
def evaluate_model(model, x_test, y_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    return predictions, y_test

# Function to provide buy/sell recommendation
def make_recommendation(predictions, y_test):
    if predictions[-1] > y_test[-1]:
        return "Buy"
    else:
        return "Sell"

# Streamlit app
st.title('Stock Price Prediction and Recommendation')
ticker = st.text_input('Enter Stock Ticker', 'AAPL')

if st.button('Analyze'):
    data = yf.download(ticker, period='1y', interval = '1d')
    st.write(data)
    # Reset the index to remove the MultiIndex
    data.reset_index(inplace=True)
    # Assuming 'data' has a MultiIndex, drop the second level of the MultiIndex
    data.columns = data.columns.droplevel(1)

    scaled_data, scaler = preprocess_data(data)
    x_train, y_train, x_test, y_test = create_datasets(scaled_data)

    model = build_model()
    model.fit(x_train, y_train, epochs=1, batch_size=1)

    predictions, y_test = evaluate_model(model, x_test, y_test, scaler)

    st.write("Predictions vs Actual")
    st.line_chart(pd.DataFrame({'Actual': y_test.flatten(), 'Predictions': predictions.flatten()}))

    recommendation = make_recommendation(predictions, y_test)
    st.write(f"Recommendation: {recommendation}")
