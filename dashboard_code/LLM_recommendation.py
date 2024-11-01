import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.tsatools import add_trend
from arch import arch_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.stock_data = self.fetch_stock_data()
        
    def fetch_stock_data(self):
        """Fetch stock data using yfinance."""
        try:
            stock = yf.Ticker(self.symbol)
            data = stock.history(period="5y")
            return data
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return None
    
    def calculate_basic_metrics(self):
        """Calculate basic stock metrics."""
        if self.stock_data is None:
            return None
        
        metrics = {
            "Current Price": self.stock_data['Close'][-1],
            "52-Week High": self.stock_data['High'].max(),
            "52-Week Low": self.stock_data['Low'].min(),
            "Average Volume": self.stock_data['Volume'].mean(),
            "Total Return (5Y)": ((self.stock_data['Close'][-1] / self.stock_data['Close'][0]) - 1) * 100
        }
        return metrics
    
    def arima_forecast(self):
        """ARIMA time series forecasting."""
        if self.stock_data is None:
            return None
        
        # Prepare data
        close_prices = self.stock_data['Close']
        
        # Fit ARIMA model
        try:
            model = ARIMA(close_prices, order=(5,1,2))
            model_fit = model.fit()
            
            # Forecast next 30 days
            forecast = model_fit.forecast(steps=30)
            return forecast
        except Exception as e:
            st.warning(f"ARIMA Forecast Error: {e}")
            return None
    
    def garch_volatility(self):
        """GARCH model for volatility prediction."""
        if self.stock_data is None:
            return None
        
        try:
            # Calculate returns
            returns = self.stock_data['Close'].pct_change().dropna()
            
            # Fit GARCH model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            model_fit = model.fit()
            
            # Get volatility forecast
            forecast = model_fit.forecast(horizon=30)
            return forecast
        except Exception as e:
            st.warning(f"GARCH Volatility Error: {e}")
            return None
    
    def lstm_prediction(self):
        """LSTM Neural Network for stock price prediction."""
        if self.stock_data is None:
            return None
        
        try:
            # Prepare data
            close_prices = self.stock_data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(close_prices)
            
            # Create sequences
            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:(i + seq_length), 0])
                    y.append(data[i + seq_length, 0])
                return np.array(X), np.array(y)
            
            seq_length = 60
            X, y = create_sequences(scaled_prices, seq_length)
            
            # Reshape for LSTM input
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(seq_length, 1)),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            # Make predictions
            last_sequence = scaled_prices[-seq_length:]
            last_sequence = last_sequence.reshape((1, seq_length, 1))
            predicted_scaled = model.predict(last_sequence)
            predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
            
            return predicted_price
        except Exception as e:
            st.warning(f"LSTM Prediction Error: {e}")
            return None
    
    def random_forest_prediction(self):
        """Random Forest regression for stock price prediction."""
        if self.stock_data is None:
            return None
        
        try:
            # Prepare features
            data = self.stock_data.copy()
            data['Day'] = range(len(data))
            
            # Select features
            features = ['Day', 'Volume']
            X = data[features]
            y = data['Close']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Predict future price
            last_day = X.iloc[-1]['Day'] + 30
            future_features = pd.DataFrame({
                'Day': [last_day],
                'Volume': [data['Volume'].mean()]
            })
            prediction = rf_model.predict(future_features)[0]
            
            return prediction
        except Exception as e:
            st.warning(f"Random Forest Prediction Error: {e}")
            return None
    
    def generate_recommendation(self):
        """Generate stock recommendation based on multiple analyses."""
        if self.stock_data is None:
            return "Unable to generate recommendation due to data fetch error."
        
        # Collect predictions from different models
        predictions = {
            "ARIMA": self.arima_forecast(),
            "LSTM": self.lstm_prediction(),
            "Random Forest": self.random_forest_prediction()
        }
        
        # Current price
        current_price = self.stock_data['Close'][-1]
        
        # Analyze predictions
        pred_values = [p for p in predictions.values() if p is not None]
        
        if not pred_values:
            return "Insufficient data to generate recommendation."
        
        avg_prediction = np.mean(pred_values)
        
        # Recommendation logic
        if avg_prediction > current_price * 1.1:
            return f"**BUY** recommendation\n\n*Justification:* Predicted price ({avg_prediction:.2f}) is significantly higher than current price ({current_price:.2f}). Multiple models suggest potential price appreciation."
        elif avg_prediction < current_price * 0.9:
            return f"**SELL** recommendation\n\n*Justification:* Predicted price ({avg_prediction:.2f}) is lower than current price ({current_price:.2f}). Models indicate potential price depreciation."
        else:
            return f"**HOLD** recommendation\n\n*Justification:* Predicted price ({avg_prediction:.2f}) is close to current price ({current_price:.2f}). Limited upside or downside potential."

def plot_stock_performance(data):
    """Create a comprehensive stock performance visualization."""
    plt.figure(figsize=(12, 6))
    
    # Price trend
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], label='Closing Price')
    plt.title(f'Stock Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Volume trend
    plt.subplot(2, 1, 2)
    plt.bar(data.index, data['Volume'], label='Trading Volume', alpha=0.5)
    plt.title('Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    
    plt.tight_layout()
    return plt

def main():
    st.title("🚀 Comprehensive Stock Analysis Tool")
    
    # Sidebar for input
    st.sidebar.header("Stock Analysis Parameters")
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
    
    # Main analysis button
    if st.sidebar.button("Analyze Stock"):
        # Fetch and validate stock data
        analyzer = StockAnalyzer(stock_symbol)
        
        if analyzer.stock_data is not None:
            # Performance Visualization
            st.header(f"Stock Performance: {stock_symbol}")
            performance_plot = plot_stock_performance(analyzer.stock_data)
            st.pyplot(performance_plot)
            
            # Performance Analysis
            st.subheader("Performance Analysis")
            metrics = analyzer.calculate_basic_metrics()
            if metrics:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Price", f"${metrics['Current Price']:.2f}")
                    st.metric("52-Week High", f"${metrics['52-Week High']:.2f}")
                with col2:
                    st.metric("52-Week Low", f"${metrics['52-Week Low']:.2f}")
                    st.metric("Average Volume", f"{metrics['Average Volume']:,.0f}")
                st.metric("Total 5-Year Return", f"{metrics['Total Return (5Y)']:.2f}%")
            
            # Machine Learning Predictions
            st.header("Machine Learning Predictions")
            
            # Display individual model predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ARIMA Forecast")
                arima_pred = analyzer.arima_forecast()
                if arima_pred is not None:
                    st.write(f"Next 30 Days Forecast: {arima_pred}")
                
                st.subheader("LSTM Prediction")
                lstm_pred = analyzer.lstm_prediction()
                if lstm_pred is not None:
                    st.write(f"Predicted Price: ${lstm_pred:.2f}")
            
            with col2:
                st.subheader("GARCH Volatility")
                garch_pred = analyzer.garch_volatility()
                if garch_pred is not None:
                    st.write("Volatility Forecast Available")
                
                st.subheader("Random Forest")
                rf_pred = analyzer.random_forest_prediction()
                if rf_pred is not None:
                    st.write(f"Predicted Price: ${rf_pred:.2f}")
            
            # Final Recommendation
            st.header("Investment Recommendation")
            recommendation = analyzer.generate_recommendation()
            st.markdown(recommendation)
        else:
            st.error(f"Could not retrieve data for stock symbol: {stock_symbol}")

if __name__ == "__main__":
    main()