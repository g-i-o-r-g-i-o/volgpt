'''vol data for volgpt.ipynb
'''

import os
import glob
import warnings
import pandas as pd
import numpy as np
# import statsmodels.api as sm
import scipy.stats as stats
from scipy import stats
# from arch.unitroot import VarianceRatio
from datetime import datetime, timedelta, date

# disable warnings emitted by warnings.warn re aesthetics of post
warnings.filterwarnings('ignore')

# import HF datasets
def import_hf_data():
    bigF = pd.read_csv("bigF.csv", index_col=False, header=0, engine='python')
    bigF = bigF.drop(columns=['Unnamed: 0'])
    return bigF

# assemble time series
def hf_data(bigF):
    # Set column headings for AAPL and JPM
    bigF.columns = ["Date","Ticker","TimeBarStart","OpenBarTime","OpenBidPrice",
                "OpenBidSize","OpenAskPrice","OpenAskSize","FirstTradeTime",
                "FirstTradePrice","FirstTradeSize","HighBidTime","HighBidPrice",
                "HighBidSize","HighAskTime","HighAskPrice","HighAskSize",
                "HighTradeTime","HighTradePrice","HighTradeSize","LowBidTime",
                "LowBidPrice","LowBidSize","LowAskTime","LowAskPrice",
                "LowAskSize","LowTradeTime","LowTradePrice","LowTradeSize",
                "CloseBarTime","CloseBidPrice","CloseBidSize","CloseAskPrice",
                "CloseAskSize","LastTradeTime","LastTradePrice","LastTradeSize",
                "MinSpread","MaxSpread","CancelSize","VolumeWeightPrice",
                "NBBOQuoteCount","TradeAtBid","TradeAtBidMid","TradeAtMid",
                "TradeAtMidAsk","TradeAtAsk","TradeAtCrossOrLocked","Volume",
                "TotalTrades","FinraVolume","FinraVolumeWeightPrice",
                "UptickVolume","DowntickVolume","RepeatUptickVolume",
                "RepeatDowntickVolume","UnknownTickVolume","TradeToMidVolWeight",
                "TradeToMidVolWeightRelative","TimeWeightBid","TimeWeightAsk"]
    
    # Set a date-time index, using OpenBarTime
    bigF['DateTimeIndex'] = pd.to_datetime(bigF['Date'].astype(str)) + pd.to_timedelta(bigF['OpenBarTime'].astype(str))
    bigF = bigF.set_index('DateTimeIndex')
    
    # Set datatypes and add separate Date and Time columns in case useful later
    bigF['Ticker'] = bigF.Ticker.astype(str)
    bigF['CloseBidSize'] = bigF.CloseBidSize.astype(float)
    bigF['CloseAskSize'] = bigF.CloseAskSize.astype(float)
    bigF['CloseBidPrice'] = bigF.CloseBidPrice.astype(float)
    bigF['CloseAskPrice'] = bigF.CloseAskPrice.astype(float)
    
    # Reduce bigF to smF
    smF = bigF[['Ticker','CloseBidSize','CloseAskSize','CloseBidPrice',
                'CloseAskPrice']].copy()   
    smF['Date'] = pd.to_datetime(bigF['Date'].astype(str))
    smF['Time'] = pd.to_timedelta(bigF['OpenBarTime'].astype(str))
    smF['DateTime'] = pd.to_datetime(bigF['Date'].astype(str)) + pd.to_timedelta(bigF['OpenBarTime'].astype(str))
    
    # Compute WeightedMidPrice using the closing prices per analysis
    smF['WeightedMidPrice'] = ((smF['CloseBidSize']*smF['CloseAskPrice']) + (smF['CloseAskSize']*smF['CloseBidPrice'])) / (smF['CloseBidSize'] + smF['CloseAskSize'])
    
    # Raw returns
    AAPL_rr = smF.loc[smF['Ticker'] == "AAPL"]
    AAPL_rr = AAPL_rr['WeightedMidPrice'] - AAPL_rr['WeightedMidPrice'].shift(1)
    AAPL_rr = AAPL_rr[AAPL_rr.notna()].copy()
    AAPL_rr = AAPL_rr[AAPL_rr != 0].copy()
    JPM_rr = smF.loc[smF['Ticker'] == "JPM"]
    JPM_rr = JPM_rr['WeightedMidPrice'] - JPM_rr['WeightedMidPrice'].shift(1)
    JPM_rr = JPM_rr[JPM_rr.notna()].copy()
    JPM_rr = JPM_rr[JPM_rr != 0].copy()
    
    # Log returns
    AAPL_lr = smF.loc[smF['Ticker'] == "AAPL"]
    AAPL_lr = np.log(AAPL_lr['WeightedMidPrice'].astype(float))
    AAPL_lr = AAPL_lr - AAPL_lr.shift(1)
    AAPL_lr = AAPL_lr[AAPL_lr.notna()].copy()
    AAPL_lr = AAPL_lr[AAPL_lr != 0].copy()
    JPM_lr = smF.loc[smF['Ticker'] == "JPM"]
    JPM_lr = np.log(JPM_lr['WeightedMidPrice'].astype(float))
    JPM_lr = JPM_lr - JPM_lr.shift(1)
    JPM_lr = JPM_lr[JPM_lr.notna()].copy()
    JPM_lr = JPM_lr[JPM_lr != 0].copy()
    
    # Remove outliers
    Q1l = AAPL_lr.quantile(0.001)   
    Q3l = AAPL_lr.quantile(0.999)   
    IQl = Q3l - Q1l
    Q1r = AAPL_rr.quantile(0.001)   
    Q3r = AAPL_rr.quantile(0.999)   
    IQr = Q3r - Q1r
    AAPL_lr = AAPL_lr[~((AAPL_lr < (Q1l - 1.5 * IQl)) | (AAPL_lr > (Q3l + 1.5 * IQl)))]
    AAPL_rr = AAPL_rr[~((AAPL_rr < (Q1r - 1.5 * IQr)) | (AAPL_rr > (Q3r + 1.5 * IQr)))]
    JPM_lr = JPM_lr[~((JPM_lr < (Q1l - 1.5 * IQl)) | (JPM_lr > (Q3l + 1.5 * IQl)))]
    JPM_rr = JPM_rr[~((JPM_rr < (Q1r - 1.5 * IQr)) | (JPM_rr > (Q3r + 1.5 * IQr)))]
    
    # log returns only for models split into estimate (E=60%) and out-of-forecast (F=40%)
    AAPL = AAPL_lr.to_numpy(copy=True)
    JPM = JPM_lr.to_numpy(copy=True)
    aaplE = AAPL[0:269618,]
    aaplF = AAPL[269618:,]
    jpmE = JPM[0:200363,]
    jpmF = JPM[200363:,]
    aaplE = aaplE[:,np.newaxis]
    aaplF = aaplF[:,np.newaxis]
    jpmE = jpmE[:,np.newaxis]
    jpmF = jpmF[:,np.newaxis]

    return (bigF, smF, AAPL_rr, JPM_rr, AAPL_lr, JPM_lr, aaplE, aaplF, jpmE, 
            jpmF)



