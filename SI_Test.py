import pandas as pd

test = pd.read_csv('SI.csv')
print(test.columns)
si = pd.DataFrame()
test[['Sector', 'Last Name']] = test['Line'].str.split(' ', expand=True)
