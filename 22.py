import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np

# Step 1: Define BSE SENSEX stock symbols
bse_sensex_stocks = [
    "RELIANCE.BO", "HDFCBANK.BO", "HINDUNILVR.BO", "INFY.BO", "ITC.BO",
    "TCS.BO", "ICICIBANK.BO", "HDFC.BO", "SBIN.BO", "BHARTIARTL.BO",
    "KOTAKBANK.BO", "LT.BO", "ASIANPAINT.BO", "MARUTI.BO", "SUNPHARMA.BO",
    "BAJFINANCE.BO", "WIPRO.BO", "TITAN.BO", "NTPC.BO", "HCLTECH.BO",
    "JSWSTEEL.BO", "TATAMOTORS.BO", "POWERGRID.BO", "ONGC.BO", "CIPLA.BO",
    "ADANIGREEN.BO", "BAJAJFINSV.BO", "HINDALCO.BO", "GRASIM.BO", "DRREDDY.BO",
    "BHARATFORG.BO", "SHREECEM.BO", "HEROMOTOCO.BO", "ULTRACEMCO.BO"
]

# Step 2: Fetch data for each stock
data = []
for symbol in bse_sensex_stocks:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        # Fetch historical data for the last trading day
        history = stock.history(period="1d")
        
        if not history.empty and 'Open' in history and 'Close' in history:
            percent_change = ((history['Close'].iloc[-1] - history['Open'].iloc[0]) / history['Open'].iloc[0]) * 100
        else:
            percent_change = np.nan  # Assign NaN if no data

        data.append({
            'Symbol': symbol,
            'Name': info.get('longName', ''),
            'Sector': info.get('sector', 'Unknown'),
            'Industry': info.get('industry', 'Unknown'),
            'Market Cap': info.get('marketCap', 0),
            'Percent Change': percent_change
        })
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

df = pd.DataFrame(data)

# Step 3: Create treemap visualization
fig = px.treemap(df, 
                 path=['Sector', 'Industry', 'Name'],
                 values='Market Cap',
                 color='Percent Change',
                 color_continuous_scale=['red', 'gray', 'green'],
                 range_color=[-5, 5],
                 hover_data=['Symbol', 'Percent Change'],
                 title='BSE SENSEX Stocks - Treemap of Market Cap and Percentage Change')

# Show the treemap plot
fig.show()

# Step 4: Create pie chart visualization for percentage change
fig_pie = px.pie(df,
                 names='Name',  # Use company names for the pie slices
                 values='Percent Change',  # Use percentage change for slice sizes
                 hover_data=['Symbol', 'Market Cap'],  # Additional data to show on hover
                 title='BSE SENSEX Stocks - Percentage Change',
                 color='Percent Change',  # Color by percentage change
                 color_continuous_scale=['red', 'gray', 'green'],
                 range_color=[-5, 5])  # Adjust this range based on your data

# Show the pie chart
fig_pie.show()