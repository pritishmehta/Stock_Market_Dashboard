import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

data = yf.download('AAPL', period = '5y')
data = yf.download('AAPL', period = '5y')
# Reset the index to remove the MultiIndex
data.reset_index(inplace=True)
# Assuming 'data' has a MultiIndex, drop the second level of the MultiIndex
data.columns = data.columns.droplevel(1)
st.write(data)
