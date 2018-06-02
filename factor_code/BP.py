# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:03:17 2018

@author: XQZhu
"""

exec(open(r'E:\FT_Users\XQZhu\stocks_backtest\prerun.py').read())
import sys
sys.path.append(r'E:\FT_Users\XQZhu\stocks_backtest\self_lib') # 自定义的函数
import pandas as pd
import database_api as dbi
import data_clean as dc

equity = dbi.get_stocks_data('equity_selected_balance_sheet', 
                             ['tot_shrhldr_eqy_excl_min_int'],
                             '2005-01-01', '2018-03-01')
fdmt = dbi.get_stocks_data('equity_fundamental_info',
                           ['type_st', 'wind_indcd', 'cap', 'listdate'],
                           '2005-01-01', '2018-03-01')
fdmt = dc.st_list(fdmt) # 将含有ST标记和上市不满一年的标记为1
data = pd.merge(fdmt, equity, on=['stkcd', 'trd_dt'], how='left')
data = data.groupby('stkcd').ffill().dropna() # 按股票向前填充
data = data.groupby('stkcd').resample('M', on='trd_dt').last() # 取每月最后的数据
data = data[(data.type_st == 0) & (data.year_1 == 0)] # 去除ST和上市不满一年的数据
bpd = data.loc[:, ['stkcd', 'trd_dt', 'wind_indcd', 'cap', 
                   'tot_shrhldr_eqy_excl_min_int']]
bpd['BP_raw'] = bpd.tot_shrhldr_eqy_excl_min_int / (10000 * bpd.cap)
b_raw = bpd.BP_raw.groupby(by='trd_dt').describe()
# 前三个月样本量较小，删除前三个月的数据
bpd.drop(pd.to_datetime(['2005-01-31', '2005-02-28', '2005-03-31']), level=1, inplace=True)
bpd = dc.clean(bpd, 'BP', by='trd_dt') # 去极值、标准化、行业中性化
bpd2 = bpd.set_index(['trd_dt', 'stkcd'])
signal_input = bpd2[['BP_neu']]
price_input = pd.DataFrame(close_price_post.stack(), columns=['close_price_post'])
signal_input.index.names = price_input.index.names
test_data = price_input.join(signal_input, how='left') # 将因子值和后复权价格合并
test_data = test_data.groupby(level=1).ffill().dropna() # 根据股票向前填充
signal_analysis(test_data['BP_neu'].unstack(), test_data['close_price_post'].unstack())

store = pd.HDFStore('test.h5') # 将测试数据保存到 test.h5
store['BP_neu'] = signal_input
store['close_price_post'] = price_input
store['test_data'] = test_data
store.close()

