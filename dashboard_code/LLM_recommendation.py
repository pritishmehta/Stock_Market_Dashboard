import streamlit as st
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import newsapi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Initialize News API
newsapi = newsapi.NewsApiClient(api_key='f958536b80ef4db0ab133be499c8bd21')

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Simplified ML Model (Replace with a trained model on stock data)
def simple_ml_model(prediction_data):
    # Placeholder for a real ML model
    if prediction_data['Sentiment'][0] > 0.05:
        return "Buy"
    elif prediction_data['Sentiment'][0] < -0.05:
        return "Sell"
    else:
        return "Hold"

def app():
    st.title("Stock Recommendation App")
    
    # User Input
    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    
    if st.button("Analyze"):
        try:
            # Fetch Stock Data
            stock_data = yf.Ticker(ticker)
            hist = stock_data.history(period="7d")
            
            # News Sentiment Analysis
            news = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy')
            sentiments = [sia.polarity_scores(article['title'])['compound'] for article in news['articles'][:5]]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Simplified Candlestick Pattern Analysis (Demonstrative)
            if hist.iloc[-1]['Close'] > hist.iloc[-2]['Close']:
                pattern = "Upward Trend"
            else:
                pattern = "Downward Trend"
            
            # ML Model Prediction (Simplified)
            prediction_data = pd.DataFrame({'Sentiment': [avg_sentiment]}, index=[0])
            recommendation = simple_ml_model(prediction_data)
            
            # Display Results
            st.subheader(f"Analysis for {ticker}:")
            st.write(f"**News Sentiment:** {avg_sentiment:.2f}")
            st.write(f"**Candlestick Pattern:** {pattern}")
            st.write(f"**ML Recommendation:** {recommendation}")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
