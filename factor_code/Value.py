# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:42:06 2018

@author: XQZhu
"""
import sys
sys.path.append(r'E:\FT_Users\XQZhu\stocks_backtest\self_lib') # 自定义的函数
import pandas as pd
import factor_test as ft
import matplotlib.pyplot as plt

store = pd.HDFStore('test_data2.h5')
retn_1m = store['retn_1m']
retn_1m_zz500 = store['retn_1m_zz500']

factors = ['BP', 'EP_TTM', 'SP_TTM', 'DP', 'OCFP_TTM', 'NCFP_TTM', 'PEG_TTM']
#factors = ['NCFP_TTM', 'PEG_TTM', 'DP']
BTIC = pd.DataFrame(index=store['btic_des_index'])
Beta = pd.DataFrame(index=store['btic_m_index'])
IC = pd.DataFrame(index=store['btic_m_index'])
Annual = pd.DataFrame(index=store['layer_des_columns'])
Sharpe = pd.DataFrame(index=store['layer_des_columns'])
Rela_IR = pd.DataFrame(index=store['layer_des_columns'])
for factor in factors:
    signal_input = store[factor + '_neu']
    test_data = ft.data_join(retn_1m, signal_input)
    btic_m = ft.btic_reg(test_data)
    btic_m.rename(columns={'ic': factor})
    btic_des_t = ft.btic_des(btic_m, factor)
    BTIC = BTIC.join(btic_des_t)
    Beta = Beta.join(btic_m.beta)
    Beta.rename(columns={'beta': factor}, inplace=True)
    IC = IC.join(btic_m.ic)
    IC.rename(columns={'ic': factor}, inplace=True)
    
    layer_des = ft.layer_test(test_data, retn_1m_zz500, quantile=5)[0]
    Annual = Annual.join(layer_des.loc['annual', :])
    Annual.rename(columns={'annual': factor}, inplace=True)
    Sharpe = Sharpe.join(layer_des.loc['sharpe', :])
    Sharpe.rename(columns={'sharpe': factor}, inplace=True)
    Rela_IR = Rela_IR.join(layer_des.loc['rela_retn_IR', :])
    Rela_IR.rename(columns={'rela_retn_IR': factor}, inplace=True)
store.close()
    
xz = range(len(Beta))
xn = Beta.index.strftime('%Y-%m')
plt.figure(figsize=(10, 5))
Beta_c = 1 + Beta.cumsum()
for i in Beta.columns:
    plt.plot(xz, Beta_c[i], label=i)
plt.legend()
plt.xlim(xz[0] - 1, xz[-1] + 1)
plt.ylim(Beta_c.min().min() - 0.05, Beta_c.max().max() + 0.05)
plt.xticks(xz[0:-1:12], xn[0:-1:12])
plt.title('Cumulated Return of Value Factor')

plt.figure(figsize=(10, 5))
quantile = 5
factor_num = len(Annual.columns)
xz = range(quantile * factor_num)
for i in range(factor_num):
    plt.bar(xz[5 * i:5*(i + 1)], Annual.iloc[:5, i])
plt.xlim(- 1, quantile * factor_num)
plt.ylim(0, Annual.max().max() + 0.01)
xn = Annual.columns.tolist()
plt.xticks(xz[quantile//2:-1:5], xn)
plt.title('Annual Return of Value Factors')

IC_corr = IC.corr()
    
    