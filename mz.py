

import pandas as pd
import numpy as np
import statsmodels.api as sm
from io import StringIO
from datetime import datetime

def perform_mz_regression(data, column_names=None):
    if column_names is None:
        column_names = ['Ticker', 'CloseBidSize', 'CloseAskSize', 'CloseBidPrice', 'CloseAskPrice', 'WeightedMidPrice', 'rr', 'lr']

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

    data_io = StringIO(data)
    df = pd.read_csv(data_io, header=None, names=column_names)

    invalid_rows = []
    for index, row in df.iterrows():
        if not check_date_format(row['DateTime']) or \
           not all(check_numeric(val) for val in row[['CloseBidSize', 'CloseAskSize', 'CloseBidPrice', 'CloseAskPrice', 'WeightedMidPrice', 'rr', 'lr']].values):
            invalid_rows.append(index)

    df_clean = df.drop(invalid_rows)

    def mz_regression(y, x):
        X = sm.add_constant(x)
        model = sm.OLS(y, X)
        results = model.fit()
        return results

    df_clean['WeightedMidPrice'] = pd.to_numeric(df_clean['WeightedMidPrice'])

    rr_results = mz_regression(df_clean['rr'], df_clean['WeightedMidPrice'])
    lr_results = mz_regression(df_clean['lr'], df_clean['WeightedMidPrice'])

    return df, df_clean, rr_results, lr_results, invalid_rows
