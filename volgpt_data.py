import os
import glob
import pandas as pd
import numpy as np
from scipy import stats

def high_frequency_data():
    path = 'allfiles' # import HF data (AAPL, JPM), being 563 daily files
    all_files = glob.glob(os.path.join(path,"**/*.csv.gz"))
    np_array_list = []
    for file_ in all_files:
        df = pd.read_csv(file_,index_col=None, header=0) # read the csv's
        np_array_list.append(df.values) # convert to numpy array

    comb_np_array = np.vstack(np_array_list) # Create bit array
    df_data = pd.DataFrame(comb_np_array) # Create dataframe

    # Set column headings
    df_data.columns = ["Date","Ticker","TimeBarStart","OpenBarTime","OpenBidPrice","OpenBidSize", "OpenAskPrice","OpenAskSize","FirstTradeTime","FirstTradePrice","FirstTradeSize","HighBidTime","HighBidPrice","HighBidSize","HighAskTime","HighAskPrice","HighAskSize","HighTradeTime","HighTradePrice","HighTradeSize","LowBidTime","LowBidPrice","LowBidSize","LowAskTime","LowAskPrice","LowAskSize","LowTradeTime","LowTradePrice","LowTradeSize","CloseBarTime","CloseBidPrice","CloseBidSize","CloseAskPrice","CloseAskSize","LastTradeTime","LastTradePrice","LastTradeSize","MinSpread","MaxSpread","CancelSize","VolumeWeightPrice","NBBOQuoteCount","TradeAtBid","TradeAtBidMid","TradeAtMid","TradeAtMidAsk","TradeAtAsk","TradeAtCrossOrLocked","Volume","TotalTrades","FinraVolume","FinraVolumeWeightPrice","UptickVolume","DowntickVolume","RepeatUptickVolume","RepeatDowntickVolume","UnknownTickVolume","TradeToMidVolWeight","TradeToMidVolWeightRelative","TimeWeightBid","TimeWeightAsk"]

    # Set datatypes for columns used to compute weighted mid-price
    df_data['CloseBidSize'] = df_data['CloseBidSize'].astype(float)
    df_data['CloseAskSize'] = df_data['CloseAskSize'].astype(float)
    df_data['CloseBidPrice'] = df_data['CloseBidPrice'].astype(float)
    df_data['CloseAskPrice'] = df_data['CloseAskPrice'].astype(float)

    # Reduce number of columns for tractability
    # NB: it may be helpful to use the other columns in the future to explore whether more data can be used to improve the model
    df_data_full = df_data.copy()
    df_data = df_data_full[['Ticker','CloseBidSize','CloseAskSize','CloseBidPrice','CloseAskPrice']].copy()   
    df_data['DateTime'] = pd.to_datetime(df_data_full['Date'].astype(str)) + pd.to_timedelta(df_data_full['OpenBarTime'].astype(str))
    
    # Compute WeightedMidPrice using the closing prices per analysis in my high-frequency-data post
    df_data['WeightedMidPrice'] = ((df_data['CloseBidSize']*df_data['CloseAskPrice']) + (df_data['CloseAskSize']*df_data['CloseBidPrice'])) / (df_data['CloseBidSize'] + df_data['CloseAskSize'])

    # Raw returns
    AAPL_rr = df_data.loc[df_data['Ticker'] == "AAPL"]
    AAPL_rr = AAPL_rr['WeightedMidPrice'] - AAPL_rr['WeightedMidPrice'].shift(1)
    JPM_rr = df_data.loc[df_data['Ticker'] == "JPM"]
    JPM_rr = JPM_rr['WeightedMidPrice'] - JPM_rr['WeightedMidPrice'].shift(1)

    # Merge AAPL_rr and JPM_rr with df_data
    df_data = df_data.merge(AAPL_rr.to_frame(name='AAPL_rr'), left_index=True, right_index=True, how='left')
    df_data = df_data.merge(JPM_rr.to_frame(name='JPM_rr'), left_index=True, right_index=True, how='left')

    # Log returns
    AAPL_lr = df_data.loc[df_data['Ticker'] == "AAPL"]
    AAPL_lr = np.log(AAPL_lr['WeightedMidPrice'].astype(float))
    AAPL_lr = AAPL_lr - AAPL_lr.shift(1)
    JPM_lr = df_data.loc[df_data['Ticker'] == "JPM"]
    JPM_lr = np.log(JPM_lr['WeightedMidPrice'].astype(float))
    JPM_lr = JPM_lr - JPM_lr.shift(1)

    # Append log returns as additional columns to df_data
    df_data = pd.concat([df_data, AAPL_lr.rename('AAPL_lr'), JPM_lr.rename('JPM_lr')], axis=1)

    # Round all numeric columns to 2 decimal place, this helps with the model's performance
    df_data['CloseBidSize'] = df_data['CloseBidSize'].round(2)
    df_data['CloseAskSize'] = df_data['CloseAskSize'].round(2)
    df_data['CloseBidPrice'] = df_data['CloseBidPrice'].round(2)
    df_data['CloseAskPrice'] = df_data['CloseAskPrice'].round(2)
    df_data['WeightedMidPrice'] = df_data['WeightedMidPrice'].round(2)
    df_data['AAPL_rr'] = df_data['AAPL_rr'].round(2)
    df_data['JPM_rr'] = df_data['JPM_rr'].round(2)
    df_data['AAPL_lr'] = df_data['AAPL_lr'].round(2)
    df_data['JPM_lr'] = df_data['JPM_lr'].round(2)

    df_data.fillna("UNK", inplace=True) # Replace missing values with "UNK"

    df_data = df_data.astype(str) # Convert all columns to string

    # Descriptive stats based upon raw returns
    AAPL_rr_stat = AAPL_rr[AAPL_rr.notna()].copy() # raw returns for AAPL using df_data
    AAPL_rr_stat = AAPL_rr_stat[AAPL_rr_stat != 0].copy()
    JPM_rr_stat = JPM_rr[JPM_rr.notna()].copy() # raw returns for JPM using df_data
    JPM_rr_stat = JPM_rr_stat[JPM_rr_stat != 0].copy()
    AAPL_stats = stats.describe(AAPL_rr_stat) # Descriptive statistics for AAPL
    JPM_stats = stats.describe(JPM_rr_stat) # Descriptive statistics for JPM

    # Split df_data into AAPL and JPM subsets
    df_data_AAPL = df_data.loc[df_data['Ticker'] == 'AAPL']
    df_data_JPM = df_data.loc[df_data['Ticker'] == 'JPM']
    
    return (df_data, df_data_AAPL, df_data_JPM, AAPL_rr, JPM_rr, AAPL_lr, JPM_lr, AAPL_stats, JPM_stats)



