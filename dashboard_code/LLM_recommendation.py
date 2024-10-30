import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ta.pattern import (
   candlestick_engulfing,
   candlestick_hammer,
   candlestick_inverted_hammer,
   candlestick_shooting_star,
   candlestick_hanging_man
)
from ta.utils import dropna
st.title("Stock Recommendation App")
# Get user input for the stock ticker
ticker = st.text_input("Enter a stock ticker:", "AAPL")
# Fetch the stock data
stock = yf.Ticker(ticker)
df = stock.history(period="1y")
# Analyze candlestick patterns
df['engulfing'] = df.apply(candlestick_engulfing, axis=1)
df['hammer'] = df.apply(candlestick_hammer, axis=1)
df['inverted_hammer'] = df.apply(candlestick_inverted_hammer, axis=1)
df['shooting_star'] = df.apply(candlestick_shooting_star, axis=1)
df['hanging_man'] = df.apply(candlestick_hanging_man, axis=1)
# Prepare the data for the machine learning model
X = df[['engulfing', 'hammer', 'inverted_hammer', 'shooting_star', 'hanging_man']]
y = (df['Close'].shift(-1) > df['Close']).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the machine learning model
model = LogisticRegression()
model.fit(X_train, y_train)
# Get the prediction and recommendation
prediction = model.predict(X_test)[0]
if prediction == 0:
   recommendation = "Hold"
elif prediction == 1:
   recommendation = "Buy"
else:
   recommendation = "Sell"
st.write(f"The recommendation for {ticker} is: {recommendation}")
