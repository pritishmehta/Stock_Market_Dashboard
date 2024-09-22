import pandas as pd
import datetime
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.express as px

st.title('Indian Dashboard')
ticker = st.text_input('Ticker','MSFT')
start_date = st.date_input('Start Date',datetime.date(2024,7,10))
end_date = st.date_input('End Date',datetime.date(2024,8,9))

data= yf.download(ticker, start_date, end_date)
fig = px.line(data, x=data.index, y=data['Adj Close'], title=ticker)
st.plotly_chart(fig)