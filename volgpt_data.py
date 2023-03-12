import os
import glob
import pandas as pd
import numpy as np
from scipy import stats

# import HF data (AAPL, JPM), being 563 daily files
def high_frequency_data():
    path = 'allfiles'
    all_files = glob.glob(os.path.join(path,"**/*.csv.gz"))
    np_array_list = []
    for file_ in all_files:
        df = pd.read_csv(file_,index_col=None, header=0) # read the csv's
        np_array_list.append(df.values) # convert to numpy array

    comb_np_array = np.vstack(np_array_list) # Create bit array
    df_data = pd.DataFrame(comb_np_array) # Create dataframe

    # Set column headings
    df_data.columns = ["Date","Ticker","TimeBarStart","OpenBarTime","OpenBidPrice","OpenBidSize", "OpenAskPrice","OpenAskSize","FirstTradeTime","FirstTradePrice","FirstTradeSize","HighBidTime","HighBidPrice","HighBidSize","HighAskTime","HighAskPrice","HighAskSize","HighTradeTime","HighTradePrice","HighTradeSize","LowBidTime","LowBidPrice","LowBidSize","LowAskTime","LowAskPrice","LowAskSize","LowTradeTime","LowTradePrice","LowTradeSize","CloseBarTime","CloseBidPrice","CloseBidSize","CloseAskPrice","CloseAskSize","LastTradeTime","LastTradePrice","LastTradeSize","MinSpread","MaxSpread","CancelSize","VolumeWeightPrice","NBBOQuoteCount","TradeAtBid","TradeAtBidMid","TradeAtMid","TradeAtMidAsk","TradeAtAsk","TradeAtCrossOrLocked","Volume","TotalTrades","FinraVolume","FinraVolumeWeightPrice","UptickVolume","DowntickVolume","RepeatUptickVolume","RepeatDowntickVolume","UnknownTickVolume","TradeToMidVolWeight","TradeToMidVolWeightRelative","TimeWeightBid","TimeWeightAsk"]

    # Set a date-time index, using OpenBarTime
    df_data['DateTimeIndex'] = pd.to_datetime(df_data['Date'].astype(str)) + pd.to_timedelta(df_data['OpenBarTime'].astype(str))
    df_data = df_data.set_index('DateTimeIndex')
    df_data = df_data.drop(['Date','TimeBarStart'], axis=1) # Drop original Date and TimeBarStart columns

    # Calculate the proportion of missing values and zero values for each column
    prop_missing = (df_data.isna().sum() + (df_data == 0).sum()) / len(df_data)
    prop_missing_pct = prop_missing.map('{:.4%}'.format)  # Format as a percentage to 4dp

    # Set datatypes for columns used to compute weighted mid-price
    df_data['CloseBidSize'] = df_data['CloseBidSize'].astype(float)
    df_data['CloseAskSize'] = df_data['CloseAskSize'].astype(float)
    df_data['CloseBidPrice'] = df_data['CloseBidPrice'].astype(float)
    df_data['CloseAskPrice'] = df_data['CloseAskPrice'].astype(float)

    # Compute WeightedMidPrice using the closing prices per analysis in my high-frequency-data post
    df_data['WeightedMidPrice'] = ((df_data['CloseBidSize']*df_data['CloseAskPrice']) + (df_data['CloseAskSize']*df_data['CloseBidPrice'])) / (df_data['CloseBidSize'] + df_data['CloseAskSize'])

    # Raw returns
    AAPL_rr = df_data.loc[df_data['Ticker'] == "AAPL"]
    AAPL_rr = AAPL_rr['WeightedMidPrice'] - AAPL_rr['WeightedMidPrice'].shift(1)
    AAPL_rr = AAPL_rr[AAPL_rr.notna()].copy()
    AAPL_rr = AAPL_rr[AAPL_rr != 0].copy()
    JPM_rr = df_data.loc[df_data['Ticker'] == "JPM"]
    JPM_rr = JPM_rr['WeightedMidPrice'] - JPM_rr['WeightedMidPrice'].shift(1)
    JPM_rr = JPM_rr[JPM_rr.notna()].copy()
    JPM_rr = JPM_rr[JPM_rr != 0].copy()

    # Merge AAPL_rr and JPM_rr with df_data
    df_data = df_data.merge(AAPL_rr.to_frame(name='AAPL_rr'), left_index=True, right_index=True, how='left')
    df_data = df_data.merge(JPM_rr.to_frame(name='JPM_rr'), left_index=True, right_index=True, how='left')

    # Log returns
    AAPL_lr = df_data.loc[df_data['Ticker'] == "AAPL"]
    AAPL_lr = np.log(AAPL_lr['WeightedMidPrice'].astype(float))
    AAPL_lr = AAPL_lr - AAPL_lr.shift(1)
    AAPL_lr = AAPL_lr[AAPL_lr.notna()].copy()
    AAPL_lr = AAPL_lr[AAPL_lr != 0].copy()
    JPM_lr = df_data.loc[df_data['Ticker'] == "JPM"]
    JPM_lr = np.log(JPM_lr['WeightedMidPrice'].astype(float))
    JPM_lr = JPM_lr - JPM_lr.shift(1)
    JPM_lr = JPM_lr[JPM_lr.notna()].copy()
    JPM_lr = JPM_lr[JPM_lr != 0].copy()

    # Append log returns as additional columns to df_data
    df_data = pd.concat([df_data, AAPL_lr.rename('AAPL_lr'), JPM_lr.rename('JPM_lr')], axis=1)

    # Descriptive statistics
    AAPL_stats = stats.describe(AAPL_rr)
    JPM_stats = stats.describe(JPM_rr)

    # returns plots see volgpt-data.py

    return (df_data, prop_missing_pct, AAPL_rr, JPM_rr, AAPL_lr, JPM_lr, AAPL_stats, JPM_stats)



