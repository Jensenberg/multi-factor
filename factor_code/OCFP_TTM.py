# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:03:48 2018

@author: XQZhu
"""

exec(open(r'E:\FT_Users\XQZhu\stocks_backtest\prerun.py').read())
import sys
sys.path.append(r'E:\FT_Users\XQZhu\stocks_backtest\self_lib') # 自定义的函数
import pandas as pd
import database_api as dbi
import data_merge as dm
import data_clean as dc

cash_flow = dbi.get_stocks_data('equity_selected_cashflow_sheet_q', 
                               ['net_cash_flows_oper_act'],
                               '2004-01-01', '2018-03-01')
cash_flow['oper_cf_ttm'] = cash_flow.groupby('stkcd')[['net_cash_flows_oper_act']].apply(
        lambda x: x.rolling(4, min_periods=4).sum())

store = pd.HDFStore('test_data.h5')
fdmt = store['fundamental_info']
store.close()

data = dm.factor_merge(fdmt, cash_flow)
data = data.loc[:, ['stkcd', 'trd_dt', 'wind_indcd', 'cap', 'oper_cf_ttm']]
data['OCFP_TTM_raw'] = data['oper_cf_ttm'] / (10000 * data['cap'])
ocf_raw = data['OCFP_TTM_raw'].groupby(level=1).describe()
data.drop(pd.to_datetime(['2005-01-31', '2005-02-28', '2005-03-31']), level=1, inplace=True)
data = dc.clean(data, 'OCFP_TTM')
signal_input, test_data = dm.test_merge(data, 'OCFP_TTM_neu', close_price_post)
dm.test_result(test_data, 'OCFP_TTM_neu', 'close_price_post')

store = pd.HDFStore('test_data.h5')
store['OCFP_TTM_neu'] =signal_input
store.close()
