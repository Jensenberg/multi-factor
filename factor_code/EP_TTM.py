# -*- coding: utf-8 -*-
"""
Created on Thu May 17 09:47:08 2018

@author: XQZhu
"""

exec(open(r'E:\FT_Users\XQZhu\stocks_backtest\prerun.py').read())
import sys
sys.path.append(r'E:\FT_Users\XQZhu\stocks_backtest\self_lib') # 自定义的函数
import pandas as pd
import database_api as dbi
import data_merge as dm
import data_clean as dc

net_profit = dbi.get_stocks_data('equity_selected_income_sheet_q',
                                 ['net_profit_excl_min_int_inc'],
                                 '2004-01-01', '2018-03-01')
net_profit['net_profit_emi_ttm'] = net_profit.groupby('stkcd')[['net_profit_excl_min_int_inc']].apply(
        lambda x: x.rolling(4, min_periods=4).sum())

store = pd.HDFStore('test_data.h5')
fdmt = store['fundamental_info']
store.close()

data = dm.factor_merge(fdmt, net_profit)
data = data.loc[:, ['stkcd', 'trd_dt', 'wind_indcd', 'cap', 'net_profit_emi_ttm']]
data['EP_TTM_raw'] = data['net_profit_emi_ttm'] / (10000 * data['cap'])
e_raw = data['EP_TTM_raw'].groupby(level=1).describe()
data.drop(pd.to_datetime(['2005-01-31', '2005-02-28', '2005-03-31']), level=1, inplace=True)
data = dc.clean(data, 'EP_TTM')
signal_input, test_data = dm.test_merge(data, 'EP_TTM_neu', close_price_post)
dm.test_result(test_data, 'EP_TTM_neu', 'close_price_post')

store = pd.HDFStore('test_data.h5')
store['EP_TTM_neu'] = signal_input
store.close()
