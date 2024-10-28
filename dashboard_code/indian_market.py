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
    st.write('Temp')
    def fetch_india_gdp_data():
        """Fetches India's GDP data from World Bank API"""
        base_url = "http://api.worldbank.org/v2/country/IND/indicator/NY.GDP.MKTP.KD.ZG"
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
            df.columns = ['Year', 'GDP Growth (%)']
            return df
        
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data: {e}")
            return None

    def create_gdp_chart(df, chart_type='line'):
        """Creates either a line or bar chart based on user selection"""
        if chart_type == 'line':
            fig = px.line(df, 
                        x='Year', 
                        y='GDP Growth (%)',
                        title='India GDP Growth Rate (Annual %)')
            fig.update_traces(mode='lines+markers')
        else:
            fig = px.bar(df,
                        x='Year',
                        y='GDP Growth (%)',
                        title='India GDP Growth Rate (Annual %)')
        
        fig.update_layout(
            template='plotly_white',
            hovermode='x',
            showlegend=False,
            height=500
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig

    def calculate_metrics(df):
        """Calculates key metrics for the dashboard"""
        latest_growth = df['GDP Growth (%)'].iloc[-1]
        avg_5year = df['GDP Growth (%)'].tail(5).mean()
        highest_growth = df['GDP Growth (%)'].max()
        lowest_growth = df['GDP Growth (%)'].min()
        
        return latest_growth, avg_5year, highest_growth, lowest_growth
    # Header
    st.title("India GDP Growth Analysis")
    st.markdown("Interactive dashboard showing India's GDP growth trends and analysis")
    
    # Fetch data
    with st.spinner("Fetching latest GDP data..."):
        df = fetch_india_gdp_data()
    
    if df is not None:
        # Sidebar controls
        st.sidebar.header("Visualization Controls")
        chart_type = st.sidebar.selectbox(
            "Select Chart Type",
            options=['line', 'bar'],
            format_func=lambda x: x.title() + " Chart"
        )
        
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max()))
        )
        
        # Filter data based on year range
        filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        
        # Calculate metrics
        latest_growth, avg_5year, highest_growth, lowest_growth = calculate_metrics(filtered_df)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Latest Growth Rate", f"{latest_growth:.1f}%")
        with col2:
            st.metric("5-Year Average", f"{avg_5year:.1f}%")
        with col3:
            st.metric("Highest Growth", f"{highest_growth:.1f}%")
        with col4:
            st.metric("Lowest Growth", f"{lowest_growth:.1f}%")
        
        # Display chart
        st.plotly_chart(create_gdp_chart(filtered_df, chart_type), use_container_width=True)
        
    else:
        st.error("Failed to fetch GDP data. Please try again later.")g