import streamlit as st
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Set up the Streamlit app
st.title("Stock Recommendation App")

# Define a function to fetch historical stock data
def fetch_historical_data(ticker, period):
    data = yf.download(ticker, period=period)
    return data

# Define a function to perform sentiment analysis on news
def sentiment_analysis(news):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(news)
    return sentiment

# Define a function to build and train a neural network
def build_neural_network(X_train, y_train):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model

# Define a function to make predictions using the neural network
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Fetch historical data for a given stock
ticker = st.text_input("Enter stock ticker")
period = st.selectbox("Select period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
data = fetch_historical_data(ticker, period)

# Perform sentiment analysis on news
news = st.text_input("Enter news article")
sentiment = sentiment_analysis(news)

# Prepare data for neural network
data['Sentiment'] = sentiment['compound']
X = data.drop(['Close'], axis=1)
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train neural network
model = build_neural_network(X_train, y_train)

# Make predictions using neural network
predictions = make_predictions(model, X_test)

# Display results
st.write(" Historical Data:")
st.write(data)
st.write("Sentiment Analysis:")
st.write(sentiment)
st.write("Neural Network Predictions:")
st.write(predictions)

# Explain the analysis
st.write("The historical data shows the past performance of the stock. The sentiment analysis of the news article provides an indication of the market sentiment towards the stock. The neural network predictions are based on the historical data and sentiment analysis, and provide a forecast of the stock's future performance.")
