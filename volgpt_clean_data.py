import pandas as pd
import numpy as np
import statsmodels.api as sm
from io import StringIO
from datetime import datetime


def clean_data(text_data, column_names=None):
    if column_names is None:
        column_names = ['DateTimeIndex', 'Ticker', 'CloseBidSize', 'CloseAskSize', 'CloseBidPrice', 'CloseAskPrice', 'WeightedMidPrice', 'rr', 'lr']

    # Data cleaning
    data_io = StringIO(text_data)
    df = pd.read_csv(data_io, header=None, names=column_names)

    # check date format
    def check_date_format(date_str):
        if pd.isna(date_str):
            return False
        try:
            datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            return True
        except ValueError:
            return False

    # Alternative date format checks
    # def check_date_format(date_str):
    #     if pd.isna(date_str):
    #         return False
    #     try:
    #         dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    #         min_date = datetime(2000, 1, 1)  # for tractability, I set ranges outside of the actual data range
    #         max_date = datetime(2050, 12, 31)
    #         if dt < min_date or dt > max_date:
    #             return False
    #         return True
    #     except ValueError:
    #         return False

    # Another alternative date format check
    # def check_date_format(date_str):
    #     if pd.isna(date_str):
    #         return False
    #     try:
    #         dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    #         min_date = pd.Timestamp.min.to_pydatetime()
    #         max_date = pd.Timestamp.max.to_pydatetime()
    #         if dt < min_date or dt > max_date:
    #             return False
    #         return True
    #     except ValueError:
    #         return False

    def check_numeric(value):
        if pd.isna(value):
            return False
        try:
            float(value)
            return True
        except ValueError:
            return False

    data_io = StringIO(text_data)
    df = pd.read_csv(data_io, header=None, names=column_names)

    invalid_rows = []
    for index, row in df.iterrows():
        if not check_date_format(row['DateTimeIndex']) or \
           not all(check_numeric(val) for val in row[['CloseBidSize', 'CloseAskSize', 'CloseBidPrice', 'CloseAskPrice', 'WeightedMidPrice', 'rr', 'lr']].values):
            invalid_rows.append(index)

    df_clean = df.drop(invalid_rows)

    # Data type conversions
    df_clean['DateTimeIndex'] = df_clean['DateTimeIndex'].apply(pd.to_datetime)
    df_clean['Ticker'] = df_clean['Ticker'].astype(str)
    df_clean['CloseBidSize'] = pd.to_numeric(df_clean['CloseBidSize'])
    df_clean['CloseAskSize'] = pd.to_numeric(df_clean['CloseAskSize'])
    df_clean['CloseBidPrice'] = pd.to_numeric(df_clean['CloseBidPrice'])
    df_clean['CloseAskPrice'] = pd.to_numeric(df_clean['CloseAskPrice'])
    df_clean['WeightedMidPrice'] = pd.to_numeric(df_clean['WeightedMidPrice'])
    df_clean['rr'] = pd.to_numeric(df_clean['rr'])
    df_clean['lr'] = pd.to_numeric(df_clean['lr'])

    # print("Outputs from clean_data function: ")
    # print("-----------------------------"), print()

    # print("Original DataFrame: ")
    # print(df)

    # print("Cleaned DataFrame: ")
    # print(df_clean)

    # print("Invalid Rows: ")
    # print(invalid_rows)

    return df, df_clean, invalid_rows