import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.express as px

us_market = st.Page(
    page="dashboard_code/US_market.py",
    title = "US Market",
    default = True
)

indian_market = st.Page(
    page='dashboard_code/indian_market.py',
    title='Indian Market'
)

crypto = st.Page(
    page='dashboard_code/crypto_market.py',
    title='Crypto Market'
)

pg = st.navigation(pages=[us_market,indian_market,crypto])

pg.run()