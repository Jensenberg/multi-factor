# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:32:39 2018

@author: XQZhu
"""

import sys
sys.path.append(r'E:\FT_Users\XQZhu\stocks_backtest\self_lib') # 自定义的函数
import pandas as pd
import database_api as dbi
import data_clean as dc
import data_merge as dm
import factor_test as ft

cash_flow = dbi.get_stocks_data('equity_selected_cashflow_sheet_q', 
                               ['net_incr_cash_cash_equ'],
                               '2004-01-01', '2018-03-01')
cash_flow['ncf_ttm'] = cash_flow.groupby('stkcd')[['net_incr_cash_cash_equ']].apply(
        lambda x: x.rolling(4, min_periods=4).sum())

store = pd.HDFStore('test_data.h5')
fdmt = store['fundamental_info']
retn_1m = store['retn_1m']
retn_1m_zz500 = store['retn_1m_zz500']
store.close()

data = dm.factor_merge(fdmt, cash_flow)
data = data.loc[:, ['stkcd', 'trd_dt', 'wind_indcd', 'cap', 'ncf_ttm']]
data['NCFP_TTM_raw'] = data['ncf_ttm'] / (10000 * data['cap'])
ncf_raw = data['NCFP_TTM_raw'].groupby(level=1).describe()
data.drop(pd.to_datetime(['2005-01-31', '2005-02-28', '2005-03-31']), level=1, inplace=True)
data = dc.clean(data, 'NCFP_TTM')
data = data.set_index(['trd_dt', 'stkcd']) 
data.index.names = ['trade_date', 'stock_ID']
signal_input = data[['NCFP_TTM_neu']]
test_data = ft.data_join(retn_1m, signal_input)
btic_des, btic_m = ft.btic(test_data, 'NCFP_TTM')
layer_des = ft.layer_result(test_data, retn_1m_zz500, 'NCFP_TTM')
