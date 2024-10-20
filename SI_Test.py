import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Set up Selenium webdriver
options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)

# Navigate to the webpage
driver.get("https://www.livemint.com/market/market-stats/stocks-varun-beverages-share-price-nse-bse-s0003351")

# Wait for the table to load
driver.implicitly_wait(10)

# Extract the table data using pandas
table_data = pd.read_html(driver.page_source)[0]

# Print the table data
print(table_data)

# Perform SWOT analysis (this part will require manual analysis and input)
# ...

# Close the webdriver
driver.quit()