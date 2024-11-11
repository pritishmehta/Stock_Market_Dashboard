import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Function to load data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to prepare data
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    return scaled_data

# Function to create and train model
def create_model(data):
    x = []
    y = []
    for i in range(60, len(data)):
        x.append(data[i-60:i, 0])
        y.append(data[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    return model, x_test, y_test

# Function to make predictions
def make_predictions(model, x_test):
    predictions = model.predict(x_test)
    return predictions

# Function to evaluate model
def evaluate_model(y_test, predictions):
    accuracy = np.sqrt(np.mean((predictions - y_test) ** 2))
    return accuracy

# Streamlit app
st.title('Stock Price Prediction using LSTM')

st.sidebar.header('Select Stock and Date Range')
ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2022-02-26'))

if st.sidebar.button('Load Data'):
    data = load_data(ticker, start_date, end_date)
    st.write(data.head())

if st.sidebar.button('Prepare Data'):
    scaled_data = prepare_data(data)
    st.write(scaled_data)

if st.sidebar.button('Create and Train Model'):
    model, x_test, y_test = create_model(scaled_data)
    st.write('Model Created and Trained')

if st.sidebar.button('Make Predictions'):
    predictions = make_predictions(model, x_test)
    st.write(predictions)

if st.sidebar.button('Evaluate Model'):
    accuracy = evaluate_model(y_test, predictions)
    st.write('Model Accuracy: ', accuracy)

# Stock worth buying
if st.sidebar.button('Stock Worth Buying'):
    if accuracy < 10:
        st.write('Stock is worth buying')
    else:
        st.write('Stock is not worth buying')
