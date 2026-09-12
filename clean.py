import pandas as pd

# Load the raw data
data = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the useful columns and rename them
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Check the balance between spam and ham
print("\nLabel counts:\n", data['label'].value_counts())

# Preview the cleaned data
print("\nCleaned data preview:")
print(data.head())
