import pandas as pd

data = pd.read_csv('spam.csv', encoding='latin-1')
print(data.head())
print(data.shape)
print(data.columns)