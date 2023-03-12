# refactored code from ChatGPT:

import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
sns.set()


class HFDataAnalysis:
    def __init__(self, path):
        self.path = path
        self.all_files = glob.glob(os.path.join(self.path, "**/*.csv.gz"))
        self.df_data = None
        
    def read_data(self):
        np_array_list = []
        for file_ in self.all_files:
            df = pd.read_csv(file_, index_col=None, header=0) # read the csv's
            np_array_list.append(df.values) # convert to numpy array

        comb_np_array = np.vstack(np_array_list) # Create bit array
        self.df_data = pd.DataFrame(comb_np_array) # Create dataframe

        # Set column headings
        self.df_data.columns = ["Date","Ticker","TimeBarStart","OpenBarTime","OpenBidPrice","OpenBidSize",
                                "OpenAskPrice","OpenAskSize","FirstTradeTime","FirstTradePrice","FirstTradeSize",
                                "HighBidTime","HighBidPrice","HighBidSize","HighAskTime","HighAskPrice","HighAskSize",
                                "HighTradeTime","HighTradePrice","HighTradeSize","LowBidTime","LowBidPrice","LowBidSize",
                                "LowAskTime","LowAskPrice","LowAskSize","LowTradeTime","LowTradePrice","LowTradeSize",
                                "CloseBarTime","CloseBidPrice","CloseBidSize","CloseAskPrice","CloseAskSize",
                                "LastTradeTime","LastTradePrice","LastTradeSize","MinSpread","MaxSpread","CancelSize",
                                "VolumeWeightPrice","NBBOQuoteCount","TradeAtBid","TradeAtBidMid","TradeAtMid",
                                "TradeAtMidAsk","TradeAtAsk","TradeAtCrossOrLocked","Volume","TotalTrades","FinraVolume",
                                "FinraVolumeWeightPrice","UptickVolume","DowntickVolume","RepeatUptickVolume",
                                "RepeatDowntickVolume","UnknownTickVolume","TradeToMidVolWeight","TradeToMidVolWeightRelative",
                                "TimeWeightBid","TimeWeightAsk"]
        
        # Set a date-time index, using OpenBarTime
        self.df_data['DateTimeIndex'] = pd.to_datetime(self.df_data['Date'].astype(str)) + \
                                        pd.to_timedelta(self.df_data['OpenBarTime'].astype(str))
        self.df_data = self.df_data.set_index('DateTimeIndex')
        self.df_data = self.df_data.drop(['Date','TimeBarStart'], axis=1) # Drop original Date and TimeBarStart columns
    
    def calculate_proportions_missing_and_zero(self):
        # Calculate the proportion of missing values and zero values for each column
        prop_missing = (self.df_data.isna().sum() + (self.df_data == 0).sum()) / len(self.df_data)
        prop_missing_pct = prop_missing.map('{:.4%}'.format)  # Format as a percentage to 4dp
        print(prop_missing_pct)

    def set_column_datatypes(self):
        # Set datatypes for columns used to compute weighted mid-price
        self.df_data['CloseBidSize'] = self.df_data['CloseBidSize'].astype(float)
        self.df_data['CloseAskSize'] = self.df_data['CloseAskSize'].astype(float)
        self.df






