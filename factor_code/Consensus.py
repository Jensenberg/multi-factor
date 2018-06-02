# -*- coding: utf-8 -*-
"""
Created on Thu May 31 09:58:09 2018

@author: XQZhu
"""

import pandas as pd
import sys
sys.path.append(r'E:\FT_Users\XQZhu\stocks_backtest\self_lib')
import database_api as dbi
import data_clean as dc
import class_test as ct

cons = dbi.get_stocks_data('equity_consensus_forecast', 
                           ['est_net_profit_FTTM', 'est_oper_revenue_FTTM', 'est_baseshare_FTTM', 
                            'rating_avg_30', 'est_price_30', 'est_price_90', 'est_price_180', 
                            'est_price_instnum_30', 'est_price_instnum_90', 'est_price_instnum_180'],
                            '2011-01-01', '2018-02-28')
cp = dbi.get_stocks_data('equity_selected_trading_data', ['close'], '2011-01-01', '2018-02-28')
cons = pd.merge(cons, cp, on=['stkcd', 'trd_dt'], how='left')
cons_m = cons.groupby('stkcd').resample('M', on='trd_dt').last()
cons_m['Fore_EPS'] = cons_m['est_net_profit_FTTM'] / cons_m['est_baseshare_FTTM'] / 10000
factors = ['rating_avg_30', 'Fore_EPS', 'est_oper_revenue_FTTM', 'est_net_profit_FTTM'] 
for factor in factors:
    cons_m[factor + '_p_1m'] = cons_m.groupby('stkcd')[factor].shift(1)
    cons_m[factor + '_p_3m'] = cons_m.groupby('stkcd')[factor].shift(3)
    cons_m[factor + '_Chge_1m'] = cons_m[factor] - cons_m[factor + '_p_1m']
    cons_m[factor + '_Chge_3m'] = cons_m[factor] - cons_m[factor + '_p_3m']

cons_m['est_net_profit_FTTM_Chge_1m_ratio'] = cons_m['est_net_profit_FTTM_Chge_1m'] /\
cons_m['est_net_profit_FTTM_p_1m']
cons_m['est_net_profit_FTTM_Chge_3m_ratio'] = cons_m['est_net_profit_FTTM_Chge_3m'] /\
cons_m['est_net_profit_FTTM_p_3m']
days = [30, 90, 180]
for day in days:
    cons_m['est_price_' + str(day) + '_retn'] = cons_m['est_price_' + str(day)] / cons_m['close'] - 1
names = []
for factor in factors[:-1]:
    chge1 = factor + '_Chge_1m'
    chge2 = factor + '_Chge_3m'
    names.extend([factor, chge1, chge2])
names2 = ['est_net_profit_FTTM', 'est_net_profit_FTTM_Chge_1m_ratio', 'est_net_profit_FTTM_Chge_3m_ratio',
          'est_price_instnum_30', 'est_price_instnum_90', 'est_price_instnum_180']
names3 = ['est_price_' + str(i) + '_retn' for i in days]
names = names + names2 + names3
consensus = cons_m.loc[:, names]
newnames =  ['Rating', 'Rating_C_1m', 'Rating_C_3m', 'Fore_EPS', 'EPS_C_1m', 'EPS_C_3m',
             'Est_Sales', 'Sales_C_1m', 'Sales_C_3m', 'Fore_Earning', 'Earning_C_1m_ratio', 
             'Earning_C_3m_ration', 'Inst_Num_30', 'Inst_Num_90', 'Inst_Num_180', 
             'Retn_30', 'Retn_90', 'Retn_180']
names_dict = {old: new for (old, new) in zip(names, newnames)}
consensus.rename(columns=names_dict, inplace=True)
consensus.index.names = ['stock_ID', 'trade_date']
store = pd.HDFStore('test_data.h5')
fdmt = store['fundamental_info']


fdmt.rename(columns={'trd_dt': 'trade_date', 'stkcd': 'stock_ID'}, inplace=True)
fdmt_m = fdmt.groupby('stock_ID').resample('M', on='trade_date').last()

Consensus = store['Consensus_index']
store.close()
Consensus = Consensus.set_index(['trade_date', 'stock_ID'])
Rating = pd.DataFrame(index=Consensus.index)
Consensus_names = ['Fore_EPS', 'EPS_C_1m', 'EPS_C_3m', 'Est_Sales', 'Sales_C_1m', 'Sales_C_3m', 
                   'Fore_Earning', 'Earning_C_1m_ratio', 'Earning_C_3m_ration', 'Retn_30', 
                   'Retn_90', 'Retn_180',  'Inst_Num_30', 'Inst_Num_90', 'Inst_Num_180', 
                   'Rating', 'Rating_C_1m', 'Rating_C_3m',]
for factor in Consensus_names:
    data = fdmt_m.join(consensus.loc[:, factor])
    data = data[(data.type_st == 0) & (data.year_1 == 0)].dropna()
    data = dc.clean(data, factor) # 对于Rating相关的三个指标，不用去极值
    data = data.set_index(['trade_date', 'stock_ID'])
    Consensus= Consensus.join(data[[factor + '_neu']], how='outer')

store = pd.HDFStore('test_data.h5')    
store['Consensus'] = Consensus
store.close()
BTIC, IC, IC_corr, Annual, Sharpe, Rela_IR = ct.class_test(Consensus_names, 'Consensus')
