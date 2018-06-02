# -*- coding: utf-8 -*-
"""
Created on Fri May 25 09:38:49 2018

@author: XQZhu
"""

import pandas as pd
import sys
sys.path.append(r'E:\FT_Users\XQZhu\stocks_backtest\self_lib')
import data_clean as dc
import class_test as ct

store = pd.HDFStore('test_data.h5')
cp = store['close_price_return_month'].iloc[:, :3]
intervals = [1, 3, 6, 12, 24] 
for i in intervals:
    cp['cp_' + str(i) + 'm'] = cp.groupby('stock_ID')['close_price_post'].shift(i)
    cp['retn_' + str(i) + 'm_p'] = cp[['close_price_post', 'cp_' + str(i) + 'm']].\
    groupby('stock_ID', group_keys=False).apply(lambda x: x['close_price_post'].\
           div(x['cp_' + str(i) + 'm'], axis='index') - 1)

retn_names = ['retn_' + str(i) + 'm_p' for i in intervals]
retn_m_p = cp.loc[:, ['trade_date', 'stock_ID'] + retn_names]

cp_d = store['close_price_post']
cp_d['retn'] = cp_d.groupby('stock_ID')['close_price_post'].apply(lambda x: x.pct_change())
retn_d = cp_d.reset_index().loc[:, ['trade_date', 'stock_ID', 'retn']]
retn_1m_p_max = retn_d.groupby('stock_ID').resample('M', on='trade_date').max()
retn_1m_p_max.rename(columns={'retn': 'retn_1m_p_max'}, inplace=True)
momentum = pd.merge(retn_m_p, retn_1m_p_max, on=['trade_date', 'stock_ID'], how='outer', left_index=True)

fdmt = store['fundamental_info']
store.close()

fdmt.rename(columns={'trd_dt': 'trade_date', 'stkcd': 'stock_ID'}, inplace=True)
fdmt_m = fdmt.groupby('stock_ID').resample('M', on='trade_date').last()
momentum_names = momentum.columns.tolist()[2:]
Momentum = store['Momentum_index']
Momentum= Momentum.set_index(['trade_date', 'stock_ID'])
for factor in momentum_names:
    data = pd.merge(fdmt_m, momentum.loc[:, ['trade_date', 'stock_ID', factor]], 
                    on=['stock_ID', 'trade_date'], left_index=True)
    data = data[(data.type_st == 0) & (data.year_1 == 0)].dropna()
    data = dc.clean(data, factor)
    data = data.set_index(['trade_date', 'stock_ID'])
    Momentum = Momentum.join(data[[factor + '_neu']], how='outer')

store['Momentum'] = Momentum
BTIC, IC, IC_corr, Annual, Sharpe, Rela_IR = ct.class_test(momentum_names, 'Momentum')
    