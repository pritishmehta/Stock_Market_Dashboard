import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np

# Step 1: Define Nifty 50 stock symbols
nifty50_symbols = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "HDFC.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
    "KOTAKBANK.NS", "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "HCLTECH.NS",
    "MARUTI.NS", "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "TITAN.NS", "SUNPHARMA.NS", "BAJAJFINSV.NS", "TECHM.NS", "ONGC.NS",
    "HDFCLIFE.NS", "NTPC.NS", "TATAMOTORS.NS", "POWERGRID.NS", "M&M.NS",
    "CIPLA.NS", "ADANIPORTS.NS", "GRASIM.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "BRITANNIA.NS", "HINDALCO.NS", "INDUSINDBK.NS", "SBILIFE.NS", "UPL.NS",
    "TATASTEEL.NS", "EICHERMOT.NS", "JSWSTEEL.NS", "COALINDIA.NS", "BPCL.NS",
    "SHREECEM.NS", "IOC.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS"
]

# Step 2: Fetch data for each stock
data = []
for symbol in nifty50_symbols:
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

# Step 3: Calculate industry-level percentage change
df['Weighted Change'] = df['Percent Change'] * df['Market Cap']
industry_changes = df.groupby('Industry').agg({
    'Weighted Change': 'sum',
    'Market Cap': 'sum'
}).reset_index()

# Calculate industry percent change
industry_changes['Industry Percent Change'] = np.where(
    industry_changes['Market Cap'] != 0,
    industry_changes['Weighted Change'] / industry_changes['Market Cap'],
    0  # Assign 0 if market cap is 0
)

# Merge industry changes back to the main dataframe
df = df.merge(industry_changes[['Industry', 'Industry Percent Change']], on='Industry', how='left')

# Filter out rows with Market Cap of zero
df = df[df['Market Cap'] > 0]

# Step 4: Create treemap visualization
fig = px.treemap(df, 
                 path=['Sector', 'Industry', 'Name'],
                 values='Market Cap',
                 color='Percent Change',
                 color_continuous_scale=['red', 'gray', 'green'],
                 range_color=[-5, 5],  # Adjust this range based on your data
                 hover_data=['Symbol', 'Percent Change', 'Industry Percent Change'],
                 title='Nifty 50 Stocks - Sectors and Industries')

# Show the plot
fig.show()

# Step 5: Create pie chart visualization for percentage change
fig_pie = px.pie(df,
                 names='Name',  # Use company names for the pie slices
                 values='Percent Change',  # Use percentage change for slice sizes
                 hover_data=['Symbol', 'Market Cap'],  # Additional data to show on hover
                 title='Nifty 50 Stocks - Percentage Change',
                 color='Percent Change',  # Color by percentage change
                 color_continuous_scale=['red', 'gray', 'green'],
                 range_color=[-5, 5])  # Adjust this range based on your data

# Show the pie chart
fig_pie.show()