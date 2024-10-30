import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ta.momentum import stoch, rsi
from ta.trend import macd, adx
from ta.volatility import bollinger_bands, average_true_range
st.title("Stock Recommendation App")
# Get user input for the stock ticker
ticker = st.text_input("Enter a stock ticker:", "AAPL")
# Fetch the stock data
stock = yf.Ticker(ticker)
df = stock.history(period="1y")
# Calculate technical indicators
df['stoch_k'], df['stoch_d'] = stoch(df['Close'])
df['rsi'] = rsi(df['Close'])
df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['Close'])
df['adx'] = adx(df['High'], df['Low'], df['Close'])
df['bb_h'], df['bb_m'], df['bb_l'] = bollinger_bands(df['Close'])
df['atr'] = average_true_range(df['High'], df['Low'], df['Close'])
# Prepare the data for the machine learning model
X = df[['stoch_k', 'stoch_d', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'adx', 'bb_h', 'bb_m', 'bb_l', 'atr']]
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
