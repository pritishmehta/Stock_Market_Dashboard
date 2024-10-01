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
search,indexes, charts, sectors, heatmap, economic_indicators, technical_analysis = st.tabs(['Search',"Index", "Charts", "Sectors", "Heatmap", "Economic Indicators","Technical Analysis"])
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
