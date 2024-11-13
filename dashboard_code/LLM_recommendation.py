import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Get user input for stock ticker
stock_ticker = input("Enter a stock ticker: ")

# Get historical data for the stock
data = yf.download(stock_ticker, start="2010-01-01", end="2022-02-26")

# Get news data for the stock
newsapi = NewsApiClient(api_key="6a04a3e5224f48b1af4938da6251d466")
news = newsapi.get_everything(q=stock_ticker, language="en")

# Create a dataframe with the historical data
df = pd.DataFrame(data)

# Add a column for the moving average
df["MA_50"] = df["Close"].rolling(window=50).mean()
df["MA_200"] = df["Close"].rolling(window=200).mean()

# Create a column for the news sentiment
df["News_Sentiment"] = 0
for i in range(len(df)):
    news_sentiment = 0
    for article in news["articles"]:
        if article["publishedAt"] < df.index[i]:
            news_sentiment += article["sentiment"]
    df.loc[i, "News_Sentiment"] = news_sentiment

# Create the training and testing data
X = df[["Open", "High", "Low", "Close", "MA_50", "MA_200", "News_Sentiment"]]
y = df["Close"].shift(-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=2)

# Make predictions
predictions = model.predict(X_test_scaled)

# Plot the results
plt.plot(y_test)
plt.plot(predictions)
plt.legend(["Actual", "Predicted"])
plt.show()

# Suggest buying or selling based on the predictions
if predictions[-1] > y_test[-1]:
    print("Buy")
else:
    print("Sell")
