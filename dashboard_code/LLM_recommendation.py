import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from transformers import pipeline

# Function to load stock data
def load_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2023-01-01")
    data.reset_index(inplace=True)
    return data

# Function to create LSTM model
def create_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to prepare data for LSTM
def prepare_data(data):
    data = data.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, training_data_len, scaled_data, dataset

# Function to predict stock prices
def predict_stock(model, scaled_data, scaler, training_data_len, dataset):
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions, y_test

# Function to analyze news sentiment
def analyze_sentiment(news):
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiments = sentiment_pipeline(news)
    return sentiments

# Streamlit app
st.title('Stock Recommendation System')
ticker = st.text_input('Enter Stock Ticker', 'AAPL')
data = load_data(ticker)
st.subheader('Historical Stock Data')
st.write(data.tail())

# Train LSTM model
x_train, y_train, scaler, training_data_len, scaled_data, dataset = prepare_data(data)
model = create_model()
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Predict stock prices
predictions, y_test = predict_stock(model, scaled_data, scaler, training_data_len, dataset)
st.subheader('Predicted vs Actual Stock Prices')
st.line_chart({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})

# News sentiment analysis
news = ["Apple's new product launch is expected to boost sales.", "Concerns over Apple's supply chain issues."]
sentiments = analyze_sentiment(news)
st.subheader('News Sentiment Analysis')
st.write(sentiments)

# Recommendation logic (simplified)
if sentiments[0]['label'] == 'POSITIVE' and predictions[-1] > y_test[-1]:
    st.write("Strong Buy Recommendation for", ticker)
else:
    st.write("No Strong Buy Recommendation for", ticker)
