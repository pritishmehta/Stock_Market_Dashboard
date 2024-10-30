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
    # Download VADER lexicon
    nltk.download('vader_lexicon', quiet=True)

    # Function to format ticker symbol for US stocks
    def format_ticker(ticker):
        # Remove any whitespace and convert to uppercase
        ticker = ticker.strip().upper()
        # Remove any existing market suffixes
        ticker = ticker.split('.')[0]
        return ticker

    # Function to get YTD data for US stocks
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

    # Function to get market indices data with enhanced error handling and diagnostics
    def get_market_indices():
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000'
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


    # Function to plot candlestick chart
    def plot_candlestick_chart(data, ticker):
        fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'])])
        
        fig.update_layout(title='Candlestick Chart',
                          yaxis_title='Price',
                          xaxis_title='Date')
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

    # Function to get financial news and analyze sentiment for US stocks
    def get_news_sentiment(ticker, stock_info):
        search_ticker = format_ticker(ticker)
        company_name = stock_info.get('longName', search_ticker)
        search_query = f"{company_name} stock NYSE NASDAQ"
        
        news_url = f'https://newsapi.org/v2/everything?q={search_query}&language=en&apiKey=f958536b80ef4db0ab133be499c8bd21'
        
        try:
            response = requests.get(news_url, timeout=5)
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
    st.title('US Stock Market Sentiment Analysis')

    st.markdown("""
    Enter a US stock symbol (e.g., AAPL, MSFT, GOOGL).
    The app will analyze both the stock data and related news sentiment.
    """)

    # Display market indices
    # Streamlit layout and market indices display
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
                    delta_color = "green"  # Positive change 
                else:
                    delta_color = "red"  # Negative change 

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
                
                # Create columns for stock info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Company Name", stock_info.get('longName', 'N/A'))
                    st.metric("Exchange", stock_info.get('exchange', 'N/A'))
                    
                with col2:
                    st.metric("Current Price", f"${stock_info.get('currentPrice', 0):,.2f}")
                    st.metric("Market Cap", f"${stock_info.get('marketCap', 0)/1e9:,.2f}B")
                    
                with col3:
                    st.metric("52 Week High", f"${stock_info.get('fiftyTwoWeekHigh', 0):,.2f}")
                    st.metric("52 Week Low", f"${stock_info.get('fiftyTwoWeekLow', 0):,.2f}")
                
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
    col1, col2 = st.columns(2)
    with col1:
        start_date_index = st.date_input('Start Date (Index)', datetime.date(2023,1,1), key='start_date_index')
    with col2:
        end_date_index = st.date_input('End Date (Index)', datetime.date(2024,10,25), key='end_date_index')

    # Define default stocks
    default_stocks = ['^NDX', '^GSPC', '^RUT', '^DJI', 'GC=F', 'SI=F']
    for i in default_stocks:
        data = yf.download(i,start=start_date_index,end=end_date_index)
        # Reset the index to remove the MultiIndex
        data.reset_index(inplace=True)
        
        # Assuming 'data' has a MultiIndex, drop the second level of the MultiIndex
        data.columns = data.columns.droplevel(1)
                # Create candlestick trace
        fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
        
        fig.update_layout(title='Candlestick Chart',
                          yaxis_title='Price',
                          xaxis_title='Date')
        st.plotly_chart(fig, use_container_width=True)
    
# In your main Streamlit app:
with charts:
    Gainers, Losers = st.tabs(["Gainers", "Losers"])
    with Gainers:
        st.write('Gainers')
        gainers = pd.read_html('https://finance.yahoo.com/markets/stocks/gainers/')
        df_1 = gainers[0]
        # Drop the unwanted columns
        df_1 = df_1.drop(columns=['Volume', 'Avg Vol (3M)', 'Market Cap', 'P/E Ratio (TTM)', '52 Wk Change %', '52 Wk Range','Day Chart'])
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
                            # Reset the index to remove the MultiIndex
                            data.reset_index(inplace=True)
                            
                            # Assuming 'data' has a MultiIndex, drop the second level of the MultiIndex
                            data.columns = data.columns.droplevel(1)
                            if not data.empty:
                                # Create candlestick trace
                                fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'])])
                                
                                fig.update_layout(title='Candlestick Chart',
                                                  yaxis_title='Price',
                                                  xaxis_title='Date')
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
        df_1 = df_1.drop(columns=['Volume', 'Avg Vol (3M)', 'Market Cap', 'P/E Ratio (TTM)', '52 Wk Change %', '52 Wk Range','Day Chart'])
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
                                                        # Reset the index to remove the MultiIndex
                            data.reset_index(inplace=True)
                            
                            # Assuming 'data' has a MultiIndex, drop the second level of the MultiIndex
                            data.columns = data.columns.droplevel(1)
                            if not data.empty:
                                # Create candlestick trace
                                fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'])])
                                
                                fig.update_layout(title='Candlestick Chart',
                                                  yaxis_title='Price',
                                                  xaxis_title='Date')
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
    
