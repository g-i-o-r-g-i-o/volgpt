import pandas as pd
import numpy as np
import statsmodels.api as sm
from io import StringIO
from datetime import datetime

# Check for errors in data and remove invalid rows

# Raw data
data = pred

# Column names
column_names = ['Ticker', 'CloseBidSize', 'CloseAskSize', 'CloseBidPrice', 'CloseAskPrice', 'DateTime', 'WeightedMidPrice', 'rr', 'lr']

# Create DataFrame, check formats, iterate through the dataframe, identify and remove invalid rows
data_io = StringIO(data)
df = pd.read_csv(data_io, header=None, names=column_names)

def check_date_format(date_str):
    if pd.isna(date_str):
        return False
    try:
        datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        return False

def check_numeric(value):
    if pd.isna(value):
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False

# Check for errors and drop invalid rows
invalid_rows = []
for index, row in df.iterrows():
    if not check_date_format(row['DateTime']) or \
       not all(check_numeric(val) for val in row[['CloseBidSize', 'CloseAskSize', 'CloseBidPrice', 'CloseAskPrice', 'WeightedMidPrice', 'rr', 'lr']].values):
        invalid_rows.append(index)

df_clean = df.drop(invalid_rows)

# Mincer-Zarnowitz (MZ) regressions
def mz_regression(y, x):
    X = sm.add_constant(x)  # Add constant to the independent variable
    model = sm.OLS(y, X)  # Create the OLS model
    results = model.fit()  # Fit the model
    return results

# Convert 'WeightedMidPrice' column to numeric
df_clean['WeightedMidPrice'] = pd.to_numeric(df_clean['WeightedMidPrice'])

# Perform MZ regressions on 'rr' and 'lr' columns using df_clean
rr_results = mz_regression(df_clean['rr'], df_clean['WeightedMidPrice'])
lr_results = mz_regression(df_clean['lr'], df_clean['WeightedMidPrice'])

# Print results

print("Original DataFrame:")
print(df)

print("\nCleaned DataFrame:")
print(df_clean)

print("MZ Regression for rr:")
print(rr_results.summary())

print("\nMZ Regression for lr:")
print(lr_results.summary())