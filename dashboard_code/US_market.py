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
# Set page config to wide mode
st.set_page_config(layout="wide")

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

st.title('US Market Dashboard')
search,indexes, charts, sectors, heatmap, economic_indicators, technical_analysis = st.tabs(['Search',"Index", "Charts", "Sectors", "Heatmap", "Economic Indicators","Technical Analysis"])
with search:
    nltk.download('vader_lexicon', quiet=True)

# Function to get YTD data
    def get_ytd_data(ticker_symbol):
        # Create a Ticker object
        ticker = yf.Ticker(ticker_symbol)
        
        # Get the current date
        current_date = datetime.date.today()
        
        # Create a datetime object for January 1st of the current year
        start_of_year = datetime.datetime(current_date.year, 1, 1)
        
        # Fetch the YTD data
        ytd_data = ticker.history(start=start_of_year, end=current_date)
        
        return ytd_data

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
            yaxis_title='Price',
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
        stock_data = yf.Ticker(ticker)
        return stock_data.info

    # Function to get financial news and analyze sentiment
    def get_news_sentiment(ticker):
        news_url = f'https://newsapi.org/v2/everything?q={ticker}&language=en&apiKey=f958536b80ef4db0ab133be499c8bd21'  # Replace with your actual API key
        
        try:
            response = requests.get(news_url)
            news_data = response.json()

            articles = []
            if 'articles' in news_data:
                for article in news_data['articles'][:10]:  # Limit to 10 articles for better visualization
                    title = article.get('title', '')
                    description = article.get('description', '')
                    url = article.get('url', '')

                    title_sentiment = analyze_sentiment(title)
                    description_sentiment = analyze_sentiment(description)

                    articles.append({
                        'title': title,
                        'title_sentiment': title_sentiment,
                        'description': description,
                        'description_sentiment': description_sentiment,
                        'url': url
                    })

            return articles

        except requests.RequestException as e:
            st.error(f"Error fetching news data: {e}")
            return []

    # Function to create a sentiment bar graph
    def create_sentiment_graph(articles):
        title_sentiments = [article['title_sentiment'] for article in articles if article['title_sentiment'] is not None]
        description_sentiments = [article['description_sentiment'] for article in articles if article['description_sentiment'] is not None]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(1, len(title_sentiments) + 1)),
            y=title_sentiments,
            name='Title Sentiment',
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            x=list(range(1, len(description_sentiments) + 1)),
            y=description_sentiments,
            name='Description Sentiment',
            marker_color='red'
        ))
        
        fig.update_layout(
            title='Sentiment Analysis of News Articles',
            xaxis_title='Article Number',
            yaxis_title='Sentiment Score',
            yaxis=dict(range=[-1, 1]),
            barmode='group',
            height=600,
            width=500,  # Set a fixed width for the graph
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)  # Adjust margins to maximize space
        )
        
        return fig

    # Helper function to format sentiment score
    def format_sentiment(sentiment):
        return f"{sentiment:.2f}" if sentiment is not None else "N/A"

    # Streamlit layout

    st.title('Stock Sentiment Analysis')

    ticker = st.text_input("Enter Ticker")
    if ticker:
        data = get_ytd_data(ticker)
        st.write(data.tail(5))  # Display last 5 days of data
        fig = plot_candlestick_chart(data, ticker)
        st.plotly_chart(fig, use_container_width=True)
    st.header('News Analysis')
    if st.button('Analyze'):
        # Get stock information
        stock_info = get_stock_info(ticker)
        st.subheader('Stock Information')
        st.write(f"Company Name: {stock_info.get('longName', 'N/A')}")
        st.write(f"Current Price: ${stock_info.get('currentPrice', 'N/A')}")
        st.write(f"52 Week High: ${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.write(f"52 Week Low: ${stock_info.get('fiftyTwoWeekLow', 'N/A')}")

        # Get news sentiment
        articles = get_news_sentiment(ticker)

        # Create two columns with adjusted widths
        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            st.subheader('News Articles and Sentiment Analysis')
            for i, article in enumerate(articles):
                st.write(f"**Article {i+1}**")
                st.write(f"**Title:** [{article['title']}]({article['url']})")
                st.write(f"Title Sentiment: {format_sentiment(article['title_sentiment'])}")
                st.write(f"Description: {article['description']}")
                st.write(f"Description Sentiment: {format_sentiment(article['description_sentiment'])}")
                st.write("---")

        with col2:
            st.subheader('Sentiment Graph')
            sentiment_graph = create_sentiment_graph(articles)
            st.plotly_chart(sentiment_graph, use_container_width=True)

with indexes:
    col1, col2 = st.columns(2)
    with col1:
        start_date_index = st.date_input('Start Date (Index)', datetime.date(2024, 1, 10), key='start_date_index')
    with col2:
        end_date_index = st.date_input('End Date (Index)', datetime.date.today(), key='end_date_index')

    # Define default stocks
    default_stocks = ['^NDX', '^GSPC', '^RUT', '^DJI', 'GC=F', 'SI=F']

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
                        
                        data = yf.download(ticker, start=start_date_index, end=end_date_index)
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

# In your main Streamlit app:
with charts:
    Gainers, Losers = st.tabs(["Gainers", "Losers"])
    with Gainers:
        st.write('Gainers')
        gainers = pd.read_html('https://finance.yahoo.com/markets/stocks/gainers/')
        df_1 = gainers[0]
        # Drop the unwanted columns
        df_1 = df_1.drop(columns=['Volume', 'Avg Vol (3M)', 'Market Cap', 'P/E Ratio (TTM)', '52 Wk Change %', '52 Wk Range'])
        df_1 = df_1.head(10)
        st.write(df_1)
        gainers = df_1['Symbol']
        split_data = gainers.str.split(expand=True)
        split_data = pd.DataFrame(split_data)
        gainers = split_data[0].tolist()
        data = yf.download(gainers,start_date_index,end_date_index)
        num_columns = 3
        # Create rows of charts
        for i in range(0, len(gainers), num_columns):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i + j < len(gainers):
                    ticker = gainers[i + j]
                    with cols[j]:
                        try:
                            # Fetch ticker info
                            ticker_info = yf.Ticker(ticker).info
                            if 'longName' in ticker_info:
                                company_name = ticker_info['longName']
                            else:
                                # Fallback names for gold and silver futures
                                company_name = 'Gold' if ticker == 'GC=F' else 'Silver' if ticker == 'SI=F' else ticker
                            
                            data = yf.download(ticker, start=start_date_index, end=end_date_index)
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

    with Losers:
        st.write('Losers')
        losers = pd.read_html('https://finance.yahoo.com/markets/stocks/losers/')
        df_1 = losers[0]
        # Drop the unwanted columns
        df_1 = df_1.drop(columns=['Volume', 'Avg Vol (3M)', 'Market Cap', 'P/E Ratio (TTM)', '52 Wk Change %', '52 Wk Range'])
        df_1 = df_1.head(10)
        st.write(df_1)
        losers = df_1['Symbol']
        split_data = losers.str.split(expand=True)
        split_data = pd.DataFrame(split_data)
        losers = split_data[0].tolist()
        data = yf.download(losers,start_date_index,end_date_index)
        num_columns = 3
        # Create rows of charts
        for i in range(0, len(losers), num_columns):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i + j < len(losers):
                    ticker = losers[i + j]
                    with cols[j]:
                        try:
                            # Fetch ticker info
                            ticker_info = yf.Ticker(ticker).info
                            if 'longName' in ticker_info:
                                company_name = ticker_info['longName']
                            else:
                                # Fallback names for gold and silver futures
                                company_name = 'Gold' if ticker == 'GC=F' else 'Silver' if ticker == 'SI=F' else ticker
                            
                            data = yf.download(ticker, start=start_date_index, end=end_date_index)
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
            
with sectors:
    st.header("Period")
    period = st.selectbox(
        "Select period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        index=3,  # Default to "1y"
        key='period_selectbox'
    )
    summary, charts, sector_corrleations = st.tabs(["Summary", "Charts", "Sector Correlations"])
    # List of sector ETFs (move this outside of both summary and charts)
    sector_etfs = {
        'Technology': 'XLK',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Utilities': 'XLU',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC'
    }

    # Function to get data for a given ETF (move this outside of both summary and charts)
    def get_sector_data(etf, period):
        ticker = yf.Ticker(etf)
        data = ticker.history(period=period)
        # Drop unwanted columns
        data = data.drop(columns=['Dividends', 'Stock Splits'])
        return data

    # Sidebar for user input (keep this outside of both summary and charts)


    # Fetch data for each sector ETF (do this once, outside of both summary and charts)
    sector_data = {}
    for sector, etf in sector_etfs.items():
        sector_data[sector] = get_sector_data(etf, period)

    with summary:
        st.title("Sector Stock Data")

        # Display data for each sector in two columns
        cols = st.columns(2)
        for i, (sector, data) in enumerate(sector_data.items()):
            with cols[i % 2]:
                st.subheader(f"{sector} Sector")
                st.dataframe(data)

        # Option to download data as CSV
        if st.button("Download Data as CSV"):
            sector_df = pd.concat(sector_data, axis=1)
            sector_df.to_csv('sector_data.csv')
            st.write("Data for all sectors has been saved to 'sector_data.csv'")

    with charts:
        st.write('Charts')

        # Display charts for each sector in two columns
        cols = st.columns(2)
        for i, (sector, data) in enumerate(sector_data.items()):
            with cols[i % 2]:
                st.subheader(f"{sector} Sector")
                # Create candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )])
                fig.update_layout(xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig, key=f'candlestick_{sector}')  # Unique key for each chart
    with sector_corrleations:
        st.subheader("Sector Correlations")

        # Create a DataFrame with closing prices for all sectors
        closing_prices = pd.DataFrame({sector: data['Close'] for sector, data in sector_data.items()})

        # Calculate the correlation matrix
        correlation_matrix = closing_prices.corr()

        # Create a heatmap using Plotly Graph Objects
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=correlation_matrix.values,
            hoverinfo='text',
            showscale=True
        ))

        # Add annotations
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix.columns)):
                fig.add_annotation(
                    x=correlation_matrix.columns[j],
                    y=correlation_matrix.index[i],
                    text=str(round(correlation_matrix.iloc[i, j], 2)),
                    showarrow=False,
                    font=dict(color="black")
                )

        fig.update_layout(
            title=f"Sector Correlations - {period}",
            xaxis_nticks=36
        )

        st.plotly_chart(fig)

        # Explanation of the correlation matrix
        st.write("""
        This heatmap shows the correlation between different sectors based on their closing prices.
        - Values close to 1 (dark blue) indicate strong positive correlation.
        - Values close to -1 (dark red) indicate strong negative correlation.
        - Values close to 0 (white) indicate little to no correlation.
        """)

        # Option to download correlation data as CSV
        if st.button("Download Correlation Data as CSV"):
            correlation_matrix.to_csv('sector_correlations.csv')
            st.write("Correlation data has been saved to 'sector_correlations.csv'")
with heatmap:
# ... (keep all the function definitions as they are) ...

# Create a dropdown menu for index selection
    index_choice = st.selectbox(
        "Which index would you like to see?",
        ("NASDAQ 100", "S&P 500", "Dow Jones")
    )
    def get_stock_data(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if 'regularMarketChangePercent' in info:
                return info['regularMarketChangePercent']
            elif 'previousClose' in info and 'currentPrice' in info:
                previous_close = info['previousClose']
                current_price = info['currentPrice']
                change_percent = ((current_price - previous_close) / previous_close) * 100
                return change_percent
            else:
                return None
        except Exception as e:
            return None

    def list_wikipedia_nasdaq100() -> pd.DataFrame:
        # Ref: https://stackoverflow.com/a/75846060/
        url = 'https://en.m.wikipedia.org/wiki/Nasdaq-100'
        return pd.read_html(url, attrs={'id': "constituents"})[0]
    
    def list_wikipedia_sp500() -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/75846060/
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        return pd.read_html(url, attrs={'id': "constituents"})[0]

    def list_slickcharts_DJ() -> pd.DataFrame:
        url = 'https://stockanalysis.com/list/dow-jones-stocks/'
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Brave/1.68.141'
        response = requests.get(url, headers={'User-Agent': user_agent})
        df = pd.read_html(response.text, match='Symbol')[0]
        df = df.drop(columns=['Market Cap','Stock Price','Revenue'])
        return df
    if index_choice == "NASDAQ 100":
        st.title('Stock Performance Treemap - NASDAQ 100')
        
        df = list_wikipedia_nasdaq100()
        df['Change'] = df['Symbol'].apply(get_stock_data)
        df.dropna(subset=['Change'], inplace=True)
        df['Change'] = df['Change'].astype(float)
        df['Change'] = df['Change'].abs()
            
        fig = px.treemap(df, 
                        path=['GICS Sector', 'GICS Sub-Industry', 'Symbol'],  # Changed 'Ticker' to 'Symbol'
                        values='Change', 
                        color='Change', 
                        color_continuous_scale='RdYlGn', 
                        title='Stock Performance by Sector and Sub-Industry - NASDAQ 100',
                        hover_data=['Change', 'Company'])
        
        st.plotly_chart(fig)

    elif index_choice == "S&P 500":
        st.title('Stock Performance Visualization - S&P 500')
        
        df_sp500 = list_wikipedia_sp500()
        df_sp500['Change'] = df_sp500['Symbol'].apply(get_stock_data)
        df_sp500.dropna(subset=['Change'], inplace=True)
        df_sp500['Change'] = df_sp500['Change'].astype(float)
        df_sp500['Change'] = df_sp500['Change'].abs()
        df_sp500 = df_sp500[df_sp500['Change'] > 0]
        
        fig_0 = px.treemap(df_sp500, 
                        path=['GICS Sector', 'GICS Sub-Industry', 'Security'], 
                        values='Change', 
                        color='Change', 
                        color_continuous_scale='RdYlGn', 
                        title='Stock Performance by Sector and Sub-Industry - S&P 500',
                        hover_data=['Change', 'Security'])
        st.plotly_chart(fig_0)

    else:  # Dow Jones
        st.title('Stock Performance Visualization - Dow Jones')
        
        df_dj = list_slickcharts_DJ()
        df_dj['% Change'] = df_dj['Symbol'].apply(get_stock_data)
        df_dj['% Change'] = df_dj['% Change'].astype(float)
        df_dj['abs_%_Change'] = df_dj['% Change'].abs()
        
        fig_1 = px.treemap(df_dj, 
                        path=['Symbol'], 
                        values='abs_%_Change', 
                        color='% Change', 
                        color_continuous_scale='RdYlGn', 
                        title='Stock Performance - Dow Jones',
                        hover_data=['% Change','Symbol'])
        
        st.plotly_chart(fig_1)
with economic_indicators:
    
# Function to get data from Yahoo Finance
    def get_yahoo_data(tickers):
        data = yf.Ticker(ticker)
        return data.history(period="5y")

    # Function to get data from FRED
    def get_fred_data(series_id):
        data = pdr.DataReader(series_id, 'fred')
        return data

    # Streamlit app
    st.title("Economic Indicators Dashboard")

    # GDP Growth (using GDP from FRED)
    st.header("GDP Growth")
    gdp_data = get_fred_data("GDP")
    st.line_chart(gdp_data)

    # Unemployment Rates (using UE RATE from FRED)
    st.header("Unemployment Rates")
    unemployment_data = get_fred_data("UNRATE")
    st.line_chart(unemployment_data)

    # Inflation Rates
    st.header("5 year Breakeven Inflation Rates")
    inflation_data = get_fred_data("T5YIE")
    st.line_chart(inflation_data)

    # Interest Rates
    st.header("Interest Rates")
    interest_data = get_fred_data("REAINTRATREARAT10Y")
    st.line_chart(interest_data)

    st.write("Data sourced from Yahoo Finance and Federal Reserve Economic Data (FRED)")

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
    