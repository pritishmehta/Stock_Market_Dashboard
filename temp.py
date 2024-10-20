import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import datetime

st.title('Indian Stock Market Dashboard')
indexes, charts, sectors, heatmap = st.tabs(["Index", "Charts", "Sectors", "Heatmap"])
with indexes:
    nse = col1, col2 = st.columns(2)
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
    gainers, losers = st.tabs(['gainers','losers'])

import fuzzywuzzy
from fuzzywuzzy import process

# Load the SI.csv file
si_data = pd.read_csv('SI.csv', encoding='latin-1')

with gainers:
    gainers = pd.read_html('https://www.livemint.com/market/nse-top-gainers')[0]
    gainers = gainers.head(10)
    st.write(gainers)
    for index, row in gainers.iterrows():
        company_name = row['Stocks'].replace('+', '')  # Remove the '+' from the company name
        choices = si_data['Company Name'].tolist()
        best_match = process.extractOne(company_name, choices, score_cutoff=90)
        if best_match:
            ticker = si_data.loc[si_data['Company Name'] == best_match[0], 'NSE_Ticker'].values[0]
            data = yf.download(ticker, start=start_date_index, end=end_date_index)
            fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                open=data['Open'],
                                                high=data['High'],
                                                low=data['Low'],
                                                close=data['Close'])])
            fig.update_layout(title=f"Candlestick Chart for {ticker}")
            st.plotly_chart(fig)
    with losers:
        losers = pd.read_html('https://www.livemint.com/market/nse-top-losers')[0]
        losers = losers.head(10)
        st.write(losers)
        for index, row in losers.iterrows():
            company_name = row['Stocks'].replace('+', '')  # Remove the '+' from the company name
            choices = si_data['Company Name'].tolist()
            best_match = process.extractOne(company_name, choices, score_cutoff=90)
            if best_match:
                ticker = si_data.loc[si_data['Company Name'] == best_match[0], 'NSE_Ticker'].values[0]
                data = yf.download(ticker, start=start_date_index, end=end_date_index)
                fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                    open=data['Open'],
                                                    high=data['High'],
                                                    low=data['Low'],
                                                    close=data['Close'])])
                fig.update_layout(title=f"Candlestick Chart for {ticker}")
                st.plotly_chart(fig)