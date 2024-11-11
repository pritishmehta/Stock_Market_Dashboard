import requests
import json

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
api_key = 'RF4IK09DF7WCEVIT'

# URLs for NSE (Nifty 50) and BSE (Sensex) indexes
nse_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NSE:NIFTY_50&apikey={api_key}'
bse_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=BSE:SENSEX&apikey={api_key}'

# Fetch data for NSE
nse_response = requests.get(nse_url)
nse_data = nse_response.json()

# Fetch data for BSE
bse_response = requests.get(bse_url)
bse_data = bse_response.json()

# Print the latest data for NSE
st.write("NSE (Nifty 50) Latest Data:")
st.write(json.dumps(nse_data, indent=4))

# Print the latest data for BSE
st.write("BSE (Sensex) Latest Data:")
st.write(json.dumps(bse_data, indent=4))
