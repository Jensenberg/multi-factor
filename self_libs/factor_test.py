# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:10:38 2018

@author: XQZhu
"""

from scipy import stats
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt
import pandas as pd

def data_join(retn, factor_value):
    '''
    Parameters
    ==========
    retn: DataFrame
        股票收益率
    factor_value: DataFrame
        因子值
    '''
    data = retn.join(factor_value, how='left').dropna()
    data_loss = 1 - len(data) / (len(factor_value) - 3000)
    print('data loss : %.4f%%' % (data_loss * 100))
    return data

def regress(data):
    '''
    data: DataFrame
        第一列为收益率，第二列为因子值
    
    计算因子收益率、t值、秩相关系数（IC值）
    
    '''
    x = data.iloc[:, 1]
    y = data.iloc[:, 0]
    sl = stats.linregress(x, y)
    beta = sl.slope
    tvalue = sl.slope / sl.stderr
    ic_value = stats.spearmanr(data)[0]
    result = pd.DataFrame({'beta': [beta], 'tvalue': [tvalue], 'ic': [ic_value]},
                           columns=['beta', 'tvalue', 'ic'])
    return result

def btic_reg(test_data):
    '''
    在截面上回归
    '''
    beta_t_ic = test_data.groupby(level=0).apply(regress)
    beta_t_ic = beta_t_ic.reset_index().set_index('trade_date').loc[:, ['beta', 'tvalue', 'ic']]    
    return beta_t_ic

def btic_des(data, factor_name):
    '''
    data: DataFrame
        含有return, tvalue, ic
    factor_name: str
        因子名称，仅用于画图时进行标记，下同
    '''
    premium_result = {'Return Mean': data.beta.mean(),
                      'Return Std': data.beta.std(),
                      'Return T-test': stats.ttest_1samp(data.beta, 0)[0],
                      'P(t > 0)': len(data[data.tvalue > 0]) / len(data),
                      'P(|t| > 2)': len(data[abs(data.tvalue) > 2]) / len(data),
                      '|t| Mean': abs(data.tvalue).mean(),
                      'IC Mean': data.ic.mean(),
                      'IC Std': data.ic.std(),
                      'P(IC > 0)': len(data[data.ic > 0]) / len(data.ic),
                      'P(IC > 0.02)': len(data[data.ic > 0.02]) / len(data.ic),
                      'IC IR': data.ic.mean() / data.ic.std()}
    premium_result = pd.DataFrame(
            premium_result,\
            columns=['Return Mean', 'Return Std', 'Return T-test', 'P(t > 0)', 
                     'P(|t| > 2)','|t| Mean', 'IC Mean', 'IC Std', 'P(IC > 0)', 
                     'P(IC > 0.02)', 'IC IR'], index=[factor_name]).T
    return premium_result

def btic_plot(data, factor_name):
    xz = range(len(data))
    xn = data.index.strftime('%Y-%m')
    
    plt.figure(figsize=(10, 5))
    beta = data['beta']
    plt.bar(xz, beta, label='Return of ' + factor_name)
    plt.legend()
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(beta.min() - 0.005, beta.max() + 0.005)
    plt.xticks(xz[0:-1:12], xn[0:-1:12])    
    
    plt.figure(figsize=(10, 5))
    beta_c = 1 + data['beta'].cumsum()
    plt.plot(xz, beta_c, label='Cumulated Return of ' + factor_name)
    plt.legend()
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(beta_c.min() - 0.005, beta_c.max() + 0.005)
    plt.xticks(xz[0:-1:12], xn[0:-1:12])
    
    plt.figure(figsize=(10, 5))
    low = floor(beta.min() * 100)
    up = ceil(beta.max() * 100)
    bins=pd.Series(range(low, up + 1)) / 100
    plt.hist(beta, bins=bins, label='Return of ' + factor_name)
    plt.legend()
    plt.xlim(low / 100, up / 100)
    plt.xticks(bins, bins)
    
    plt.figure(figsize=(10, 5))
    t = data['tvalue']
    plt.bar(xz, t, label='T Value of Return of ' + factor_name)
    plt.legend()
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(t.min() - 1, t.max() + 1)
    plt.xticks(xz[0:-1:12], xn[0:-1:12])

    plt.figure(figsize=(10, 5))
    ic = data['ic']
    plt.bar(xz, ic, label='IC of ' + factor_name)
    plt.legend()
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(ic.min() - 0.01, ic.max() + 0.01)
    plt.xticks(xz[0:-1:12], xn[0:-1:12])

def btic(test_data, factor_name):
    '''
    Parameters
    ==========
    factor_value: DataFrame
        因子值
    factor_name: str
        因子名称
    retn: DataFrame
        股票收益率
        
    Returns
    =======
    des: DataFrame
        因子收益率与IC值的评价指标
    btic_m: DataFrame
        每一期的beta，T值，IC值
    
    返回相应的图表
    '''
    btic_m = btic_reg(test_data)
    des = btic_des(btic_m, factor_name)
    btic_plot(btic_m, factor_name)
    return des, btic_m

def drawdown(x):
    '''
    Parametes
    =========
    x: DataFrame
        净值数据
    '''
    drawdown = []
    for t in range(len(x)):
        max_t = x[:t + 1].max()
        drawdown_t = min(0, (x[t] - max_t) / max_t)
        drawdown.append(drawdown_t)
    return pd.Series(drawdown).min()

def layer_test(test_data, retn_mk, quantile=10):
    '''
    Parameters
    ==========
    test_data: DataFrame
        包含因子值、收益率
    retn_mk: DataFrame
        市场收益率数据
    quantile: int
        层数
    '''
    test_data['layer'] = test_data.groupby('trade_date', group_keys=False).apply(
             lambda x: pd.qcut(x.iloc[:, 1], quantile, labels=False))
    layer_retn = test_data.groupby(['trade_date', 'layer'])['retn_1m'].mean()
    test_data.drop('layer', axis=1, inplace=True)
    layer_retn = layer_retn.unstack()
    layer_retn['t_b'] = layer_retn.iloc[:, -1] - layer_retn.iloc[:, 0]
    layer_retn = layer_retn.join(retn_mk, how='left').dropna()
    layer_retn.rename(columns={retn_mk.columns[0]: retn_mk.columns[0][8:]}, inplace=True)
    
    nav = (1 + layer_retn).cumprod()
    hpy = nav.iloc[-1, :] - 1
    annual = (nav.iloc[-1, :]) ** (12 / len(layer_retn)) - 1
    sigma = layer_retn.std() * sqrt(12)
    sharpe = (annual - 0.036) / sigma
    max_drdw = nav.apply(drawdown)
    
    rela_retn = layer_retn.sub(layer_retn.iloc[:, -1], axis='index')
    rela_nav = (1 + rela_retn).cumprod()
    rela_annual = (rela_nav.iloc[-1, :]) ** (12 / len(rela_retn)) - 1
    rela_sigma = rela_retn.std() * sqrt(12)
    rela_retn_IR = rela_annual / rela_sigma
    rela_max_drdw = rela_nav.apply(drawdown)
    
    result = {'hpy': hpy, 
              'annual': annual, 
              'sigma': sigma,
              'sharpe': sharpe, 
              'max_drdw': max_drdw,
              'rela_annual': rela_annual,
              'rela_sigma': rela_sigma,
              'rela_retn_IR': rela_retn_IR,
              'rela_max_drdw': rela_max_drdw}
    result = pd.DataFrame(result, 
                          columns=['hpy', 'annual', 'sigma', 'sharpe', 
                                   'max_drdw', 'rela_annual', 'rela_sigma',
                                   'rela_retn_IR', 'rela_max_drdw'])
    return result.T, nav, layer_retn

def nav_plot(data, factor_name, quantile=10):
    xz = range(len(data))
    xn = data.index.strftime('%Y-%m')
    
    plt.figure(figsize=(10, 5))
    for i in range(quantile):
        plt.plot(xz, data.iloc[:, i], label=data.columns[i])
    plt.plot(xz, data['t_b'], 'r--', label='t_b')
    plt.plot(xz, data.iloc[:, -1], 'cx--', label=data.columns[-1])
    plt.legend()
    plt.title('NAV of ' + factor_name +' Portfolios')
    plt.xlim(xz[0] - 1, xz[-1] + 1)
    plt.ylim(0, data.max().max() + 1)
    plt.xticks(xz[0:-1:12], xn[0:-1:12])

def ann_bar(data, factor_name, quantile=10):    
    plt.figure(figsize=(10, 5))
    xz = range(quantile + 2)
    plt.bar(xz, data, label='Annualized Return')
    plt.legend()
    plt.xlim(- 1, quantile + 2)
    plt.ylim(data.min() - 0.01, data.max() + 0.01)
    xn = data.index.tolist()
    plt.xticks(xz, xn)
    
def layer_result(test_data, retn_mk, factor_name, quantile=10):
    '''
    Parameters
    ==========
    factor_name: str
        因子名称，仅用于画图时进行标记
    test_data: DataFrame
        包含因子值和股票收益率
    retn_mk: DataFrame
        市场指数的收益率
    quantile: int
        层数
        
    Returns:
    =======
    layer_describe: DataFrame
        分层测试的各项评价指标
    nav: DataFrame
        组合净值
    layer_retn: DataFrame:
        组合各期收益率
    返回相应图形
    '''
    layer_describe, nav, layer_retn = layer_test(test_data, retn_mk, quantile)
    nav_plot(nav, factor_name, quantile)
    ann_bar(layer_describe.loc['annual', :], factor_name, quantile)
    return layer_describe #, nav, layer_retn

