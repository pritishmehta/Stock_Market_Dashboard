import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

def extract_specific_text(url, text_pattern):
    """Extracts specific text from a web page using BeautifulSoup.

    Args:
        url (str): The URL of the web page.
        text_pattern (str or re.Pattern): The text pattern to search for.

    Returns:
        tuple: A tuple containing the extracted line and the URL.
    """

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        soup = BeautifulSoup(response.content, 'html.parser')

        if isinstance(text_pattern, str):
            text_pattern = re.compile(text_pattern)

        specific_text_element = soup.find(text=text_pattern)

        if specific_text_element:
            # Get the parent element's text, which should contain the entire line
            parent_element = specific_text_element.parent
            line_text = parent_element.get_text()
            return line_text, url
        else:
            print("Text not found.")
            return None, None

    except requests.exceptions.RequestException as e:
        print("Error fetching URL:", e)
        return None, None

# Read the CSV file
tick = pd.read_csv('stock_matching_results.csv', encoding='iso-8859-1')

# Create a list of URLs
url_list = [f"https://www.screener.in/company/{i}/" for i in tick['TICKER']]

# Initialize an empty list to store the extracted data
data = []

count = 0
for url in url_list:
    print("Processing URL:", url)
    extracted_line, extracted_url = extract_specific_text(url, "Industry")
    if extracted_line:
        data.append([extracted_line, extracted_url])
    count += 1
    if count % 50 == 0:
        print("Sleeping for 10 seconds...")
        time.sleep(25)

# Create a DataFrame from the extracted data
df = pd.DataFrame(data, columns=["Line", "URL"])

print(df)
df.to_csv('SI.csv')