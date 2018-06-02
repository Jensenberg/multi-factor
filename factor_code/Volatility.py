# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:32:57 2018

@author: XQZhu
"""

import pandas as pd
from pandas.tseries.offsets import MonthEnd
from math import sqrt
import database_api as dbi
from scipy import stats
import numpy as np
from time import time
import sys
sys.path.append(r'E:\FT_Users\XQZhu\stocks_backtest\self_lib')
import data_clean as dc
import class_test as ct

store = pd.HDFStore('test_data.h5')
cp_d = store['close_price_post']
store.close()
cp_1m_p = pd.DataFrame()
cp_d['retn'] = cp_d.groupby('stock_ID')['close_price_post'].apply(lambda x: x.pct_change())
cp_d_m = cp_d.reset_index()
cp_max = cp_d_m.loc[:, ['stock_ID', 'trade_date', 'close_price_post']].groupby(
        'stock_ID').resample('M', on='trade_date').max()
cp_max.rename(columns={'close_price_post': 'cp_max'}, inplace=True)
cp_max = cp_max.dropna(how='all')
cp_min = cp_d_m.loc[:, ['stock_ID', 'trade_date', 'close_price_post']].groupby(
        'stock_ID').resample('M', on='trade_date').min()
cp_min.rename(columns={'close_price_post': 'cp_min'}, inplace=True)
cp_min = cp_min.dropna(how='all')
cp_min_max = cp_min[['cp_1m_min']].join(cp_max['cp_1m_max'])

for i in [3, 6]:
    cp_min_max['cp_' + str(i) + 'm_min'] = cp_min_max['cp_1m_min'].groupby(\
               'stock_ID').apply(lambda x: x.rolling(i, min_periods=i).min())
    cp_min_max['cp_' + str(i) + 'm_max'] = cp_min_max['cp_1m_max'].groupby(\
               'stock_ID').apply(lambda x: x.rolling(i, min_periods=i).max())
for i in [1, 3, 6]:
    cp_min_max['HighLow_' + str(i) + 'm'] = cp_min_max[\
               'cp_' + str(i) + 'm_max'] / cp_min_max['cp_' + str(i) + 'm_min']
# 3月份只有一个数据需要去除    
HighLow_1m = cp_min_max.loc[:, ['HighLow_1m']].unstack().iloc[:, :-1].stack()  
HighLow_3m = cp_min_max.loc[:, ['HighLow_3m']].unstack().iloc[:, :-1].stack()
HighLow_6m = cp_min_max.loc[:, ['HighLow_6m']].unstack().iloc[:, :-1].stack()
HighLow1 = HighLow_1m.join(HighLow_3m, how='outer')
HighLow = HighLow1.join(HighLow_6m, how='outer')

retn =cp_d['retn'].unstack()
# 日期处理函数
#for i in range(1, 37):
#    k = - ((12 - i) // 12)
#    j = 12 - (12 - i) % 12
#    print(i, j, k)
def std_m(retn, interval, months=159):
    std = pd.DataFrame()
    for i in range(1, months - interval):
        start_year = 2005 - ((12 - i) // 12)
        start_month = 12 - (12 - i) % 12        
        start_date = '%.4d-%.2d' % (start_year, start_month)
        
        end_year = 2005 - ((12 - (i + interval)) // 12)
        end_month =  12 - (12 - (i + interval)) % 12
        end_date = '%.4d-%.2d' % (end_year, end_month)
        
        date = pd.to_datetime(end_date) + MonthEnd()
        data = retn.loc[start_date: end_date, :]
        std[date] = data.std() * (sqrt(252 / len(data)))
        
    return std.stack()

intervals = [0, 2, 5]
retn_std = {}
for interval in intervals:
    std = std_m(retn, interval)
    retn_std['Std_'+ str(interval + 1) + 'm'] = std
retn_std = pd.DataFrame(retn_std)
retn_std.index.names = ['stock_ID', 'trade_date']

amount = dbi.get_stocks_data('equity_selected_trading_data', ['amount'], 
                            '2005-01-01', '2018-03-01')    
amount = amount.set_index(['trd_dt', 'stkcd'])
amount.index.names = ['trade_date', 'stock_ID']
store = pd.HDFStore('test_data.h5')
close_price_post = store['close_price_post']
store.close()
price_amount = close_price_post.join(amount)
price = price_amount.loc[:, 'close_price_post'].unstack()
amount = price_amount.loc[:, 'amount'].unstack()

def vstd_m(price, amount, interval, months=159):
    vstd = pd.DataFrame()
    for i in range(1, months - interval):
        start_year = 2005 - ((12 - i) // 12)
        start_month = 12 - (12 - i) % 12        
        start_date = '%.4d-%.2d' % (start_year, start_month)
        
        end_year = 2005 - ((12 - (i + interval)) // 12)
        end_month =  12 - (12 - (i + interval)) % 12
        end_date = '%.4d-%.2d' % (end_year, end_month)
        
        date = pd.to_datetime(end_date) + MonthEnd()
        price_data = price.loc[start_date: end_date, :]
        price_std = price_data.std()
        amount_sum = amount.loc[start_date: end_date, :].sum()
        vstd[date] = amount_sum / price_std * sqrt(252 / len(price_data))
        
    return vstd.stack()

intervals = [0, 2, 5]
v_std = {}
for interval in intervals:
    vstd = vstd_m(price, amount, interval)
    v_std['V_Std_'+ str(interval + 1) + 'm'] = vstd
v_std = pd.DataFrame(v_std)
v_std.index.names = ['stock_ID', 'trade_date']

store = pd.HDFStore('test_data.h5')
close_price_post = store['close_price_post']
zz500 = store['zz500']
store.close()

market_retn = zz500.pct_change()
stock_market = retn.join(market_retn)

def resid_std(stock_market, stock_name, interval=11):
    r_std = {}
    for i in range(1, 148):
        start_year = 2005 - ((12 - i) // 12)
        start_month = 12 - (12 - i) % 12        
        start_date = '%.4d-%.2d' % (start_year, start_month)
        
        end_year = 2005 - ((12 - (i + interval)) // 12)
        end_month =  12 - (12 - (i + interval)) % 12
        end_date = '%.4d-%.2d' % (end_year, end_month)
        date = pd.to_datetime(end_date) + MonthEnd()
        
        reg_data = stock_market.loc[start_date: end_date, [stock_name, 'zz500']]
        reg_data = reg_data.dropna()
        if reg_data.empty:
            r_std[date] = np.nan
        else:
            sl = stats.linregress(reg_data['zz500'], reg_data[stock_name])
            res = reg_data[stock_name] - sl.slope * reg_data['zz500'] - sl.intercept
            r_std[date] = res.std() * (sqrt(252 / len(reg_data)))
            
    return pd.Series(r_std)

stock_names = retn.columns.tolist()
resid_vol = {}
t0 = time()
for i in range(1000,3565):
    stock_name = stock_names[i]
    if i % 100 == 0:
        print(i)
    resid_vol[stock_name] = resid_std(stock_market, stock_name)
time() - t0 # 432 minutes
Resid_vol = pd.DataFrame(resid_vol)
Resid_vol = Resid_vol.reset_index().set_index('index')
Resid_vol = pd.DataFrame(Resid_vol.T.stack(), columns=['Residual_Volatility'])
Resid_vol.index.names = ['stock_ID', 'trade_date']

Vol1 = HighLow.join(retn_std, how='outer')
Vol2 = Vol1.join(v_std, how='outer')
Vol = Vol2.join(Resid_vol, how='outer')

store = pd.HDFStore('test_data.h5')
fdmt = store['fundamental_info']
store.close()

fdmt.rename(columns={'trd_dt': 'trade_date', 'stkcd': 'stock_ID'}, inplace=True)
fdmt_m = fdmt.groupby('stock_ID').resample('M', on='trade_date').last()

# volatility_index = data.loc[:, ['trade_date', 'stock_ID']]
Volatility = store['Volatility_index']
Volatility = Volatility.set_index(['trade_date', 'stock_ID'])
Volatility_names = Vol.columns.tolist()
for factor in Volatility_names:
    data = fdmt_m.join(Vol.loc[:, factor])
    data = data[(data.type_st == 0) & (data.year_1 == 0)].dropna()
    data = dc.clean(data, factor)
    data = data.set_index(['trade_date', 'stock_ID'])
    Volatility = Volatility.join(data[[factor + '_neu']], how='outer')

store['Volatility'] = Volatility
ind = list(np.sort(data['wind_2'].unique()))[1:]
BTIC, IC, IC_corr, Annual, Sharpe, Rela_IR = ct.class_test(Volatility_names, 'Volatility')
