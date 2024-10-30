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
    # Download VADER lexicon
    nltk.download('vader_lexicon', quiet=True)

    # Function to format ticker symbol for Indian stocks
    def format_ticker(ticker):
        # Remove any existing .NS or .BO suffix
        ticker = ticker.replace('.NS', '').replace('.BO', '')
        # Check if it's a valid ticker format
        if ticker.isalnum():  # Basic validation
            return f"{ticker}.NS"  # Default to NSE
        return ticker
    def get_market_indices():
        indices = {
            '^NSEI': 'Nifty 50',
            '^BSESN' : 'BSE Sensex'
        }
        
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=15)  # Get last 5 trading days
        
        indices_data = {}
        for symbol, name in indices.items():
            try:
                # Download index data from Yahoo Finance
                data = yf.download(symbol, start=start_date, end=today)
                # Reset the index to remove the MultiIndex
                data.reset_index(inplace=True)
                # Assuming 'data' has a MultiIndex, drop the second level of the MultiIndex
                data.columns = data.columns.droplevel(1)
                if not data.empty:
                    latest_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    pct_change = ((latest_price - prev_price) / prev_price) * 100
                    indices_data[name] = {
                        'price': latest_price,
                        'change': pct_change
                    }
                else:
                    st.error(f"No data retrieved for {name} (symbol: {symbol}). Check symbol or data availability.")
            except Exception as e:
                st.error(f"Error fetching {name} data: {e}")
        
        return indices_data
    # Function to get YTD data for Indian stocks
    def get_ytd_data(ticker_symbol):
        try:
            # Create a Ticker object with formatted ticker
            formatted_ticker = format_ticker(ticker_symbol)
            ticker = yf.Ticker(formatted_ticker)
            
            # Get the current date
            current_date = datetime.date.today()
            
            # Create a datetime object for January 1st of the current year
            start_of_year = datetime.datetime(current_date.year, 1, 1)
            
            # Fetch the YTD data
            ytd_data = ticker.history(start=start_of_year, end=current_date)
            
            return ytd_data
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return None

    # Function to plot candlestick chart
    def plot_candlestick_chart(data, ticker):
        fig = go.Figure()

        # Add candlestick trace
        fig.add_trace(go.Candlestick(x=data.index,
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close'],
                                    name=f'{ticker} Candlestick'))

        fig.update_layout(
            title=f'{ticker} Price (Candlestick)',
            yaxis_title='Price (₹)',
            xaxis_rangeslider_visible=True,
            height=500,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

        return fig

    # Function to analyze sentiment of a text
    def analyze_sentiment(text):
        if not text:
            return None
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(text)['compound']
        return sentiment_score

    # Function to get stock information using yfinance
    def get_stock_info(ticker):
        try:
            formatted_ticker = format_ticker(ticker)
            stock_data = yf.Ticker(formatted_ticker)
            return stock_data.info
        except Exception as e:
            st.error(f"Error fetching stock information: {e}")
            return {}

    # Function to get financial news and analyze sentiment for Indian stocks
    def get_news_sentiment(ticker, stock_info):
        # Remove .NS or .BO suffix for news search
        search_ticker = ticker.replace('.NS', '').replace('.BO', '')
        
        # Add company name for better news relevance
        company_name = stock_info.get('longName', search_ticker)
        search_query = f"{company_name} stock NSE BSE India"
        
        news_url = f'https://newsapi.org/v2/everything?q={search_query}&language=en&apiKey=f958536b80ef4db0ab133be499c8bd21'
        
        try:
            response = requests.get(news_url, timeout=5)  # Added timeout
            news_data = response.json()
            articles = []
            
            if 'articles' in news_data:
                for article in news_data['articles'][:10]:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    url = article.get('url', '')
                    published_at = article.get('publishedAt', '')

                    try:
                        pub_date = datetime.datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
                    except:
                        pub_date = datetime.datetime.now()

                    title_sentiment = analyze_sentiment(title)
                    description_sentiment = analyze_sentiment(description)

                    articles.append({
                        'title': title,
                        'title_sentiment': title_sentiment,
                        'description': description,
                        'description_sentiment': description_sentiment,
                        'url': url,
                        'published_at': pub_date
                    })

                articles.sort(key=lambda x: x['published_at'], reverse=True)
                return articles

        except Exception as e:
            st.error(f"Error fetching news data: {e}")
            return []

    # Function to create a sentiment graph
    def create_sentiment_graph(articles):
        title_sentiments = [article['title_sentiment'] for article in articles if article['title_sentiment'] is not None]
        description_sentiments = [article['description_sentiment'] for article in articles if article['description_sentiment'] is not None]
        
        if not title_sentiments and not description_sentiments:
            return None
            
        fig = go.Figure()
        
        if title_sentiments:
            fig.add_trace(go.Bar(
                x=list(range(1, len(title_sentiments) + 1)),
                y=title_sentiments,
                name='Title Sentiment',
                marker_color='blue'
            ))
        
        if description_sentiments:
            fig.add_trace(go.Bar(
                x=list(range(1, len(description_sentiments) + 1)),
                y=description_sentiments,
                name='Description Sentiment',
                marker_color='red'
            ))
        
        fig.update_layout(
            title='News Sentiment Analysis',
            xaxis_title='Article Number',
            yaxis_title='Sentiment Score',
            yaxis=dict(range=[-1, 1]),
            barmode='group',
            height=600,
            width=500,
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig

    # Helper function to format sentiment score
    def format_sentiment(sentiment):
        if sentiment is None:
            return "N/A"
        elif sentiment >= 0.05:
            return f"🟢 Positive ({sentiment:.2f})"
        elif sentiment <= -0.05:
            return f"🔴 Negative ({sentiment:.2f})"
        else:
            return f"⚪ Neutral ({sentiment:.2f})"

    # Streamlit layout
    st.title('Indian Stock Market Sentiment Analysis')

    st.markdown("""
    Enter the stock symbol (e.g., RELIANCE, TCS, INFY). 
    The app will automatically append .NS for NSE stocks.
    """)
    st.subheader('Market Overview')
    indices_data = get_market_indices()
    print(indices_data)
    if indices_data:
        # Safely create columns if indices_data is not empty
        cols = st.columns(4)
        for i, (index_name, index_data) in enumerate(indices_data.items()):
            with cols[i]:
                # Determine delta color based on the change value
                if (index_data['change'] == 0).any():
                    delta_color = "off"  # No change 
                elif (index_data['change'] > 0).any():
                    delta_color = "normal"  # Positive change 
                else:
                    delta_color = "inverse"  # Negative change 

                st.metric(
                    label=index_name,
                    #for localhost
                    value=f"${index_data['price']:,.2f}",
                    delta=f"{index_data['change']:+.2f}%",
                    #for deployment
                    #value=f"${index_data['price'].iloc[0]:,.2f}",
                    #delta=f"{index_data['change'].iloc[0]:+.2f}%",
                    delta_color=delta_color
                )
    else:
        st.warning("Market indices data is unavailable.")
    # Initialize session state for loading indicator
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False

    ticker = st.text_input("Enter Stock Symbol").upper()

    if ticker:
        st.session_state.is_loading = True
        
        formatted_ticker = format_ticker(ticker)
        
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["📈 Stock Information", "📰 News & Sentiment"])
        
        with tab1:
            # Fetch and display stock data
            data = get_ytd_data(formatted_ticker)
            if data is not None and not data.empty:
                # Display basic stock information
                stock_info = get_stock_info(formatted_ticker)
                
                # Create two columns for stock info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Company Name", stock_info.get('longName', 'N/A'))
                    st.metric("Current Price", f"₹{stock_info.get('currentPrice', 0):,.2f}")
                    
                with col2:
                    st.metric("52 Week High", f"₹{stock_info.get('fiftyTwoWeekHigh', 0):,.2f}")
                    st.metric("52 Week Low", f"₹{stock_info.get('fiftyTwoWeekLow', 0):,.2f}")
                
                # Display recent price data
                st.subheader('Recent Price Data')
                st.dataframe(data.tail(5))
                
                # Display candlestick chart
                st.subheader('Price Chart')
                fig = plot_candlestick_chart(data, formatted_ticker)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to fetch stock data. Please check the ticker symbol.")
        
        with tab2:
            with st.spinner('Fetching news and analyzing sentiment...'):
                # Get news and sentiment
                articles = get_news_sentiment(formatted_ticker, stock_info)
                
                if articles:
                    # Create two columns for news and sentiment
                    col1, col2 = st.columns([0.6, 0.4])
                    
                    with col1:
                        st.subheader('Recent News Articles')
                        for i, article in enumerate(articles):
                            with st.expander(f"Article {i+1}: {article['title'][:100]}..."):
                                st.write(f"**Title Sentiment:** {format_sentiment(article['title_sentiment'])}")
                                st.write(f"**Description:** {article['description']}")
                                st.write(f"**Description Sentiment:** {format_sentiment(article['description_sentiment'])}")
                                st.write(f"**Published:** {article['published_at'].strftime('%Y-%m-%d %H:%M')}")
                                st.write(f"**Read more:** [{article['url']}]({article['url']})")
                    
                    with col2:
                        st.subheader('Sentiment Analysis')
                        sentiment_graph = create_sentiment_graph(articles)
                        if sentiment_graph:
                            st.plotly_chart(sentiment_graph, use_container_width=True)
                        
                        # Calculate and display average sentiment
                        title_sentiments = [a['title_sentiment'] for a in articles if a['title_sentiment'] is not None]
                        if title_sentiments:
                            avg_sentiment = sum(title_sentiments) / len(title_sentiments)
                            st.metric("Overall Market Sentiment", format_sentiment(avg_sentiment))
                else:
                    st.warning("No recent news articles found for this stock.")
        
        st.session_state.is_loading = False
with indexes:
    start_date_index_1 = st.date_input('Start Date (Index)', datetime.date(2024, 1, 10), key='start_date_index_1')
    end_date_index_1 = st.date_input('End Date (Index)', datetime.date.today(), key='end_date_index_1')
    # Define default stocks
    default_stocks = ['^NSEI','^BSESN']

    num_columns = 3

    # Create rows of charts
    for i in range(0, len(default_stocks), num_columns):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            if i + j < len(default_stocks):
                ticker = default_stocks[i + j]
                with cols[j]:
                    try:
                        # Fetch ticker info
                        ticker_info = yf.Ticker(ticker).info
                        if 'longName' in ticker_info:
                            company_name = ticker_info['longName']
                        else:
                            # Fallback names for gold and silver futures
                            company_name = 'Gold' if ticker == 'GC=F' else 'Silver' if ticker == 'SI=F' else ticker
                        
                        data = yf.download(ticker, start=start_date_index_1, end=end_date_index_1)
                        if not data.empty:
                            # Create candlestick chart
                            fig = go.Figure(data=[go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name=ticker)])
                            
                            fig.update_layout(
                                title=f'{company_name} ({ticker}) Price',
                                yaxis_title='Price',
                                xaxis_rangeslider_visible=True,  # This enables the rangeslider
                                height=500,  # Increased height to accommodate the rangeslider
                                width=None,
                                xaxis=dict(
                                    rangeselector=dict(
                                        buttons=list([
                                            dict(count=1, label="1m", step="month", stepmode="backward"),
                                            dict(count=6, label="6m", step="month", stepmode="backward"),
                                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                                            dict(count=1, label="1y", step="year", stepmode="backward"),
                                            dict(step="all")
                                        ])
                                    ),
                                    rangeslider=dict(visible=True),
                                    type="date"
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write(f"No data available for {company_name} ({ticker})")
                    except Exception as e:
                        st.write(f"Error fetching data for {ticker}: {str(e)}")

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

    def fetch_stock_data(ticker, start_date, end_date):
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df, stock

    def calculate_moving_averages(df):
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        return df

    def calculate_rsi(df, window=14):
        rsi_indicator = RSIIndicator(df['Close'], window=window)
        df['RSI'] = rsi_indicator.rsi()
        return df

    def calculate_macd(df):
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        return df

    def plot_stock_data(df):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.5, 0.2, 0.3])

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                    low=df['Low'], close=df['Close'], name='Candlestick'),
                    row=1, col=1)

        # Moving Averages
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='MA200', line=dict(color='red')), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram'), row=3, col=1)

        fig.update_layout(height=900, title='Stock Technical Analysis')
        return fig

    def get_fundamental_metrics(stock):
        info = stock.info
        metrics = {
            'Price-to-Earnings Ratio': info.get('trailingPE', 'N/A'),
            'Forward P/E': info.get('forwardPE', 'N/A'),
            'Dividend Yield (%)': info.get('dividendYield', 'N/A'),
            'Debt-to-Equity Ratio': info.get('debtToEquity', 'N/A'),
            'Return on Equity (%)': info.get('returnOnEquity', 'N/A'),
            'Price-to-Book Ratio': info.get('priceToBook', 'N/A'),
            'Operating Margin (%)': info.get('operatingMargins', 'N/A'),
            'Beta': info.get('beta', 'N/A')
        }
        
        # Convert to percentage where applicable
        for key in ['Dividend Yield (%)', 'Return on Equity (%)', 'Operating Margin (%)']:
            if metrics[key] != 'N/A':
                metrics[key] = f"{metrics[key]*100:.2f}%"
        
        return metrics


    st.title('Stock Analysis Dashboard')

    ticker = st.text_input('Enter Stock Ticker (e.g., AAPL)', value='AAPL')
    start_date = st.date_input('Start Date', pd.to_datetime('2023-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2023-12-31'))

    if st.button('Analyze', key='analyze_button'):
        df, stock = fetch_stock_data(ticker, start_date, end_date)
        df = calculate_moving_averages(df)
        df = calculate_rsi(df)
        df = calculate_macd(df)

        # Technical Analysis Plot
        fig = plot_stock_data(df)
        st.plotly_chart(fig, use_container_width=True)

        # Fundamental Metrics
        st.subheader('Fundamental Metrics')
        metrics = get_fundamental_metrics(stock)
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        st.table(metrics_df)

        # Recent Price Data
        st.subheader('Recent Price Data')
        st.dataframe(df.tail())
