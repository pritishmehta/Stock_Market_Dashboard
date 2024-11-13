import streamlit as st
import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
from dateutil import parser

# Set up the Streamlit app
st.title("Stock Price Prediction")

# **Data Retrieval**
def retrieve_stock_data(ticker_symbol, start_date, end_date):
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error retrieving stock data: {str(e)}")
        return None

def retrieve_news_data(api_key, ticker_symbol):
    try:
        newsapi = NewsApiClient(api_key=api_key)
        news = newsapi.get_everything(q=ticker_symbol, language="en")
        return news
    except Exception as e:
        st.error(f"Error retrieving news data: {str(e)}")
        return None

# **Data Processing**
def process_stock_data(data):
    df = pd.DataFrame(data)
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()
    return df

def process_news_data(news, df):
    df["News_Sentiment"] = 0
    for i in range(len(df)):
        news_sentiment = 0
        for article in news["articles"]:
            article_date = parser.isoparse(article["publishedAt"])
            article_date_utc = article_date.astimezone(pytz.UTC)
            df_index_date_utc = df.index[i].to_pydatetime().replace(tzinfo=pytz.UTC).astimezone(pytz.UTC)
            if article_date_utc < df_index_date_utc:
                news_sentiment += article["description"]  # Temporary sentiment scoring (TO DO: improve)
        df.loc[i, "News_Sentiment"] = news_sentiment
    return df

# **Modeling**
def create_lstm_model(X_train_scaled):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_scaled.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def train_model(model, X_train_scaled, y_train):
    try:
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=2)
        return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

# **Visualization**
def visualize_predictions(y_test, predictions):
    fig, ax = plt.subplots()
    ax.plot(y_test)
    ax.plot(predictions)
    ax.legend(["Actual", "Predicted"])
    ax.set_title("Actual vs. Predicted Stock Prices")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    st.pyplot(fig)

# **Main App**
def main():
    # User input
    ticker_symbol = st.text_input("Enter a stock ticker symbol", value="AAPL")
    api_key = "6a04a3e5224f48b1af4938da6251d466"  # Replace with your News API key
    start_date = "2010-01-01"
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    # Data retrieval
    data = retrieve_stock_data(ticker_symbol, start_date, end_date)
    news = retrieve_news_data(api_key, ticker_symbol)

    if data is not None and news is not None:
        # Data processing
        df = process_stock_data(data)
        df = process_news_data(news, df)

        # Feature scaling and splitting
        X = df[["Open", "High", "Low", "Close", "MA_50", "MA_200", "News_Sentiment"]]
