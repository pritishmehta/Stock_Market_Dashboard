import pandas as pd

data = ['s-p-bse-100-stocks-I0004', 's-p-bse-200-stocks-I0008', 's-p-bse-500-stocks-I0005',
        'nifty-smallcap-100-stocks-I0022', 'nifty-next-50-stocks-I0032', 'nifty-midcap-100-stocks-I0011',
        'nifty-bank-stocks-I0006', 's-p-bse-smallcap-stocks-I0038', 's-p-bse-midcap-stocks-I0036', 'nifty-50-stocks-I0002']

# Create an empty list to store DataFrames
dataframes = []

# Iterate through the data and read HTML tables
for extension in data:
    url = f'https://www.livemint.com/market/{extension}'
    try:
        # Read HTML tables from the URL
        tables = pd.read_html(url)

        # Check if any tables were found
        if tables:
            # Assuming you want the first table, adjust the index if needed
            df = tables[0]

            # Add a "Link" column to the DataFrame
            df['Link'] = url

            # Append the DataFrame to the list
            dataframes.append(df)
        else:
            print(f"No tables found for URL: {url}")
    except Exception as e:
        print(f"Error occurred for URL: {url}, Error: {e}")

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(dataframes, ignore_index=True)

# Remove duplicates based on "Stocks" and "Companies" columns
final_df.drop_duplicates(subset=['Stocks', 'Sector'], keep='first', inplace=True)
final_df.drop(columns=['Price','Change','%Change','Volume(CR)'],inplace=True)
# Print the final DataFrame
print(final_df)
final_df.to_csv('Sectors.csv')
