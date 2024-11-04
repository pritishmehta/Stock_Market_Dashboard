import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Function to fetch stock data
def fetch_stock_data(stock):
    data = yf.download(stock, period='1y')
    if data.empty:
        st.error("No data found for this stock. Please try again.")
        return None
    return data

# Function to prepare data for training
def prepare_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create and train LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to create and train Dense model
def create_dense_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to make predictions
def make_predictions(model, data):
    predictions = model.predict(data)
    return predictions

# Streamlit app
st.title('Stock Analysis and Prediction')

# User input
stock = st.text_input('Enter stock symbol', 'AAPL')
model_type = st.selectbox('Select model', ['LSTM', 'Dense'])

# Fetch and prepare data
data = fetch_stock_data(stock)
if data is None:
    st.stop()

scaled_data, scaler = prepare_data(data)

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Reshape data for LSTM model
if model_type == 'LSTM':
    train_data = np.reshape(train_data, (train_data.shape[0], 1, 1))
    test_data = np.reshape(test_data, (test_data.shape[0], 1, 1))
    model = create_lstm_model((1, 1))
else:
    model = create_dense_model((train_data.shape[1],))

model.fit(train_data, train_data, epochs=50, batch_size=32, verbose=1)

# Make predictions
predictions = make_predictions(model, test_data)

# Inverse transform the predictions to original scale
predictions = scaler.inverse_transform(predictions)

# Display results
st.write('Predictions:')
st.write(predictions)

# Explain analysis
st.write('Analysis:')
st.write('The model is trained on the historical closing prices of the stock.')
st.write('The predictions are made on the test data, which is the last 20% of the total data.')
st.write('The model is trying to predict the next closing price of the stock.')
st.write('You can use these predictions to decide whether to buy or sell the stock.')

# Suggest whether to buy or sell
if predictions[-1] > data['Close'].iloc[-1]:
    st.write('Suggestion: Buy')
else:
    st.write('Suggestion: Sell')
