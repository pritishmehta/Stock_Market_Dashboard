import pandas as pd
import datetime
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly as px
import pandas as pd
from yahoo_fin.stock_info import get_day_gainers, get_day_losers
from mpl_finance import candlestick_ohlc
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas_datareader as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import MACD
from ta.momentum import RSIIndicator
from fuzzywuzzy import process
import numpy as np
# Custom CSS to expand the width
st.markdown("""
<style>
.reportview-container .main .block-container {
    max-width: 95%;
    padding-top: 5rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 5rem;
}
</style>
""", unsafe_allow_html=True)
tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_National_Stock_Exchange_of_India')

# Create an empty list to store the DataFrames
dfs = []
def fuzzy_merge(df1, df2, key1, key2, threshold=90, limit=1):
    """
    :param df1: the left table to join
    :param df2: the right table to join
    :param key1: key column of the left table
    :param key2: key column of the right table
    :param threshold: how close the matches should be to return a match, based on Levenshtein distance
    :param limit: the amount of matches that will get returned, these are sorted high to low
    :return: dataframe with boths keys and matches
    """
    s = df2[key2].tolist()

    m = df1[key1].apply(lambda x: process.extract(x, s, limit=limit))
    df1['matches'] = m

    m2 = df1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
    df1['matches'] = m2

    return df1

# Iterate through the tables
for df in tables:
    # Check if the 'Symbol' column exists
    if 'Symbol' in df.columns:
        # Split the 'Symbol' column
        df[['Index', 'Ticker']] = df['Symbol'].str.split(':', expand=True)
        # Drop the 'Symbol' column
        df = df.drop('Symbol', axis=1)
        # Append the DataFrame to the list
        dfs.append(df)

# Concatenate all the DataFrames in the list
merged_df = pd.concat(dfs, ignore_index=True)

st.title('Indian Stock Market Dashboard')
search,indexes, charts, sectors, economic_indicators, technical_analysis = st.tabs(['Search',"Index", "Charts", "Sectors", "Economic Indicators","Technical Analysis"])
with search:
    st.write('Temp')
with indexes:
    col1, col2 = st.columns(2)
    with col1:
        start_date_index = st.date_input('Start Date (Index)', datetime.date(2024, 1, 10), key='start_date_index')
    with col2:
        end_date_index = st.date_input('End Date (Index)', datetime.date.today(), key='end_date_index')
    indices = ['^NSEI', '^BSESN']
    index_names = {
        '^NSEI': 'Nifty 50',
        '^BSESN': 'BSE Sensex'
    }
    data = yf.download(indices, start=start_date_index, end=end_date_index)
    for index in indices:
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                            open=data['Open'][index],
                                            high=data['High'][index],
                                            low=data['Low'][index],
                                            close=data['Close'][index])])
        fig.update_layout(title=f"Candlestick Chart for {index_names[index]}")
        st.plotly_chart(fig)

with charts:
    Gainers,Losers = st.tabs(['Gainers','Losers'])
    with Gainers:
        df = pd.read_html('https://www.financialexpress.com/market/nse-top-gainers/')[0]

        df = df.drop(columns=['LTP','Chg','Prev Close','High','Low','Volume'])
        df= df.head(10)
        df.rename(columns={'Name': 'Company name'}, inplace=True)
        df1 = fuzzy_merge(df, merged_df, 'Company name', 'Company name', threshold=90)
        df1 = df1[df1['matches'] != '']
        merged_df_1 = pd.merge(df1, merged_df, left_on='matches', right_on='Company name')
        final_df = merged_df_1[['Company name_x', 'Ticker', '%Chg']]
        st.write(final_df)

        for index, row in final_df.iterrows():
            ticker = row['Ticker'] + '.NS'
            data = yf.download(ticker, period='5y', interval='1d')

            fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                open=data['Open'],
                                                high=data['High'],
                                                low=data['Low'],
                                                close=data['Close'])])

            fig.update_layout(title=f"Candlestick Chart for {ticker}")
            st.plotly_chart(fig)
    with Losers:
        df = pd.read_html('https://www.financialexpress.com/market/nse-top-losers/')[0]

        df = df.drop(columns=['LTP','Chg','Prev Close','High','Low','Volume'])
        df= df.head(10)
        df.rename(columns={'Name': 'Company name'}, inplace=True)


        df1 = fuzzy_merge(df, merged_df, 'Company name', 'Company name', threshold=90)
        df1 = df1[df1['matches'] != '']
        merged_df_1 = pd.merge(df1, merged_df, left_on='matches', right_on='Company name')
        final_df = merged_df_1[['Company name_x', 'Ticker', '%Chg']]
        st.write(final_df)

        for index, row in final_df.iterrows():
            ticker = row['Ticker'] + '.NS'
            data = yf.download(ticker, period='5y', interval='1d')

            fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                open=data['Open'],
                                                high=data['High'],
                                                low=data['Low'],
                                                close=data['Close'])])

            fig.update_layout(title=f"Candlestick Chart for {ticker}")
            st.plotly_chart(fig)
with sectors:
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
            st.write(f"Error fetching data for {symbol}: {e}")

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
    st.plotly_chart(fig)
    # Step 5: Create pie chart visualization for market cap weightage
    # Calculate total market cap for each industry
    industry_market_cap = df.groupby('Industry')['Market Cap'].sum().reset_index()

    # Calculate total market cap for each sector
    sector_market_cap = df.groupby('Sector')['Market Cap'].sum().reset_index()

    # Create pie chart for industry market cap weightage
    fig_pie_industry_market_cap = px.pie(industry_market_cap,
                                        names='Industry',  
                                        values='Market Cap',  
                                        hover_data=['Industry'],  
                                        title='Nifty 50 Stocks - Industry Market Capitalization Weightage',
                                        color='Market Cap'  
                                        )  

    # Create pie chart for sector market cap weightage
    fig_pie_sector_market_cap = px.pie(sector_market_cap,
                                        names='Sector',  
                                        values='Market Cap',  
                                        hover_data=['Sector'],  
                                        title='Nifty 50 Stocks - Sector Market Capitalization Weightage',
                                        color='Market Cap'  
                                        )  

    # Show the pie charts
    st.plotly_chart(fig_pie_industry_market_cap)
    st.plotly_chart(fig_pie_sector_market_cap)
with economic_indicators:

    st.write("India Economic Dashboard")

    # Fetch data for a specific indicator from the World Bank API
    def fetch_data(indicator, column_name):
        base_url = f"http://api.worldbank.org/v2/country/IND/indicator/{indicator}"
        params = {
            "format": "json",
            "per_page": 100,
            "date": "2000:2024"
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()[1]
            
            df = pd.DataFrame(data)
            df['value'] = pd.to_numeric(df['value'])
            df['date'] = pd.to_numeric(df['date'])
            df = df[['date', 'value']].sort_values('date')
            df.columns = ['Year', column_name]  # Rename columns for consistency
            return df
        
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data for {column_name}: {e}")
            return None

    def create_chart(df, y_label, title, chart_type='line'):
        if chart_type == 'line':
            fig = px.line(df, x='Year', y=y_label, title=title)
            fig.update_traces(mode='lines+markers')
        else:
            fig = px.bar(df, x='Year', y=y_label, title=title)
            
        fig.update_layout(
            template='plotly_white',
            hovermode='x',
            showlegend=False,
            height=500
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig

    def calculate_metrics(df, column_name):
        latest_value = df[column_name].iloc[-1]
        avg_5year = df[column_name].tail(5).mean()
        highest_value = df[column_name].max()
        lowest_value = df[column_name].min()
        
        return latest_value, avg_5year, highest_value, lowest_value

    # Header
    st.title("India Economic Dashboard")
    st.markdown("Interactive dashboard showing India's economic indicators including GDP growth, unemployment rate, inflation, and interest rates.")

    # Fetch data
    with st.spinner("Fetching latest economic data..."):
        gdp_df = fetch_data("NY.GDP.MKTP.KD.ZG", "GDP Growth (%)")  # GDP growth
        unemployment_df = fetch_data("SL.UEM.TOTL.ZS", "Unemployment Rate (%)")  # Unemployment rate
        inflation_df = fetch_data("FP.CPI.TOTL.ZG", "Inflation Rate (%)")  # Inflation rate
        interest_df = fetch_data("FR.INR.RINR", "Interest Rate (%)")  # Real interest rate

    # Check if all data is available
    if all(df is not None for df in [gdp_df, unemployment_df, inflation_df, interest_df]):
        # Sidebar controls
        st.sidebar.header("Visualization Controls")
        chart_type = st.sidebar.selectbox(
            "Select Chart Type",
            options=['line', 'bar'],
            format_func=lambda x: x.title() + " Chart"
        )
        
        # Set year range based on the data
        year_min = int(min(gdp_df['Year'].min(), unemployment_df['Year'].min(), inflation_df['Year'].min(), interest_df['Year'].min()))
        year_max = int(max(gdp_df['Year'].max(), unemployment_df['Year'].max(), inflation_df['Year'].max(), interest_df['Year'].max()))
        
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max)
        )
        
        # Filter data based on year range
        gdp_filtered = gdp_df[(gdp_df['Year'] >= year_range[0]) & (gdp_df['Year'] <= year_range[1])]
        unemployment_filtered = unemployment_df[(unemployment_df['Year'] >= year_range[0]) & (unemployment_df['Year'] <= year_range[1])]
        inflation_filtered = inflation_df[(inflation_df['Year'] >= year_range[0]) & (inflation_df['Year'] <= year_range[1])]
        interest_filtered = interest_df[(interest_df['Year'] >= year_range[0]) & (interest_df['Year'] <= year_range[1])]

        # Display metrics and charts for each indicator
        for title, data_df, y_label in [
            ("GDP Growth Rate (Annual %)", gdp_filtered, "GDP Growth (%)"),
            ("Unemployment Rate (%)", unemployment_filtered, "Unemployment Rate (%)"),
            ("Inflation Rate (%)", inflation_filtered, "Inflation Rate (%)"),
            ("Real Interest Rate (%)", interest_filtered, "Interest Rate (%)")
        ]:
            # Calculate metrics
            latest, avg_5year, highest, lowest = calculate_metrics(data_df, y_label)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"Latest {title}", f"{latest:.1f}%")
            with col2:
                st.metric("5-Year Average", f"{avg_5year:.1f}%")
            with col3:
                st.metric("Highest", f"{highest:.1f}%")
            with col4:
                st.metric("Lowest", f"{lowest:.1f}%")
            
            # Display chart
            st.plotly_chart(create_chart(data_df, y_label, title, chart_type), use_container_width=True)

    else:
        st.error("Failed to fetch some data. Please try again later.")
with technical_analysis:
    st.title("Stock Technical Analysis Dashboard")

    # User inputs
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
    start_date = st.date_input("Start Date", datetime(2022, 1, 1))
    end_date = st.date_input("End Date", datetime.now())

    # Check if the start date is before the end date
    if start_date > end_date:
        st.error("Start date must be before end date.")
    else:
        # Fetch data
        @st.cache_data
        def load_data(ticker, start, end):
            """Loads data from Yahoo Finance."""
            data = yf.download(ticker, start=start, end=end)
            data["Date"] = data.index
            return data

        if ticker:
            data = load_data(ticker, start_date, end_date)

            if not data.empty:
                st.write(f"### {ticker} Stock Price and Volume")
                
                # Plot price chart with volume
                fig = go.Figure()

                # Candlestick chart for price data
                fig.add_trace(go.Candlestick(
                    x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Candlestick'
                ))

                # Volume bar chart
                fig.add_trace(go.Bar(
                    x=data['Date'],
                    y=data['Volume'],
                    name='Volume',
                    yaxis='y2',
                    marker=dict(color='blue', opacity=0.5)
                ))

                # Additional layout
                fig.update_layout(
                    title=f"{ticker} Price Chart with Volume",
                    yaxis_title="Price",
                    yaxis2=dict(title="Volume", overlaying="y", side="right"),
                    xaxis_title="Date",
                    template="plotly_white",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                # Technical Indicators
                st.write("### Technical Analysis Indicators")

                # Moving Average
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['SMA_200'] = data['Close'].rolling(window=200).mean()

                # RSI (Relative Strength Index)
                rsi_indicator = RSIIndicator(close=data['Close'], window=14)
                data['RSI'] = rsi_indicator.rsi()

                # MACD (Moving Average Convergence Divergence)
                macd_indicator = MACD(close=data['Close'])
                data['MACD'] = macd_indicator.macd()
                data['MACD_Signal'] = macd_indicator.macd_signal()
                data['MACD_Hist'] = macd_indicator.macd_diff()

                # Plot Moving Averages on Price Chart
                fig.add_trace(go.Scatter(
                    x=data['Date'], y=data['SMA_50'],
                    mode='lines', name='SMA 50', line=dict(color='orange')
                ))
                fig.add_trace(go.Scatter(
                    x=data['Date'], y=data['SMA_200'],
                    mode='lines', name='SMA 200', line=dict(color='purple')
                ))

                # RSI Plot
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data['Date'], y=data['RSI'],
                    mode='lines', name='RSI', line=dict(color='blue')
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")  # Overbought line
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")  # Oversold line
                fig_rsi.update_layout(title=f"{ticker} RSI (Relative Strength Index)", height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)

                # MACD Plot
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=data['Date'], y=data['MACD'],
                    mode='lines', name='MACD', line=dict(color='blue')
                ))
                fig_macd.add_trace(go.Scatter(
                    x=data['Date'], y=data['MACD_Signal'],
                    mode='lines', name='Signal Line', line=dict(color='red')
                ))
                fig_macd.add_trace(go.Bar(
                    x=data['Date'], y=data['MACD_Hist'],
                    name='MACD Histogram', marker=dict(color='gray')
                ))
                fig_macd.update_layout(title=f"{ticker} MACD", height=300)
                st.plotly_chart(fig_macd, use_container_width=True)

            else:
                st.error("No data found for the given ticker. Please check the ticker symbol and date range.")
        else:
            st.write("Enter a ticker symbol to view data.")