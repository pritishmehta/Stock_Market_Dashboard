# prompt: draft a python code to get all the news and historical data for a stock that user inputs

import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient

def get_stock_data(ticker, period="5y"):
    """
    Fetches historical stock data and news sentiment.

    Args:
        ticker: The stock ticker symbol (e.g., "AAPL").
        period: The period for historical data (default: "5y").

    Returns:
        A tuple containing:
        - historical_data: A pandas DataFrame of historical stock data.
        - sentiment: A dictionary containing sentiment scores.
        - news_articles: A list of news articles.
        Or None if an error occurs.
    """
    try:
        # Fetch historical data
        historical_data = yf.download(ticker, period=period)

        # Fetch news data (replace with your actual API key)
        news_api_key = '6a04a3e5224f48b1af4938da6251d466'  
        newsapi = NewsApiClient(api_key=news_api_key)
        news = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy')
        news_articles = news['articles']

        # Perform sentiment analysis (example using the first article's title)
        if news_articles:
          from nltk.sentiment.vader import SentimentIntensityAnalyzer
          sia = SentimentIntensityAnalyzer()
          sentiment = sia.polarity_scores(news_articles[0]['title'])
        else:
          sentiment = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}


        return historical_data, sentiment, news_articles
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    ticker_symbol = input("Enter the stock ticker symbol: ")
    stock_info = get_stock_data(ticker_symbol)

    if stock_info:
        historical_data, sentiment, news_articles = stock_info
        print("\nHistorical Data:")
        print(historical_data)
        print("\nSentiment Analysis:")
        print(sentiment)
        print("\nNews Articles:")
        for article in news_articles:
            print(article['title'])

# prompt: based on the above code use deep learning to analyse the news and historical data to recommend buying or selling the stock

import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# ... (your existing code for fetching data, sentiment analysis, etc.)

def build_and_train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1) # Reduced epochs for faster execution in this example
    return model


if __name__ == "__main__":
    ticker_symbol = input("Enter the stock ticker symbol: ")
    stock_info = get_stock_data(ticker_symbol)

    if stock_info:
        historical_data, sentiment, news_articles = stock_info

        # Data preprocessing for LSTM
        data = historical_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        # Create sequences for LSTM
        def create_sequences(data, seq_length):
            xs = []
            ys = []
            for i in range(len(data)-seq_length-1):
                x = data[i:(i+seq_length)]
                y = data[i+seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        seq_length = 50 # Example sequence length
        X_train, y_train = create_sequences(train_data, seq_length)
        X_test, y_test = create_sequences(test_data, seq_length)


        # Reshape data for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Build and train the LSTM model
        model = build_and_train_model(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test)

        # ... (rest of your code for plotting and analysis)

        # Calculate accuracy metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        st.write("Accuracy Metrics:")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"R-Squared (R2): {r2}")
        # Calculate and print accuracy (R-squared is a common measure of accuracy)
        accuracy = r2_score(y_test, predictions)
        st.write(f"Model Accuracy (R-squared): {accuracy}")

        # Make a recommendation based on the analysis (example)
        if predictions[-1][0] > historical_data['Close'].iloc[-1].any():
            recommendation = "BUY"
        else:
            recommendation = "SELL"
        st.write(f"Recommendation: {recommendation}")
