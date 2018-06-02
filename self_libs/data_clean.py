# -*- coding: utf-8 -*-
"""
Created on Wed May  2 18:32:47 2018

@author: XQZhu
"""

import pymysql
import pandas as pd
import numpy as np
from datetime import timedelta

def get_data(table, columns):    
    '''
    Parameters
    ==========
    table: strings
        查询表格
    columns: list
        查询的字段
        
    Returns
    =======
    data: DataFrame
    '''
    db = pymysql.connect('192.168.1.140', 'ftresearch', 'FTResearch', 'ftresearch')
    cur = db.cursor()
    fields = ','.join(columns)
    query = 'Select ' + fields + 'FROM ' + table
    cur.execute(query)
    data = cur.fetchall()
    data = pd.DataFrame(list(data))
    data.columns = columns
    return data

def st_list(x):    
    '''
    Returns
    =======
    将含有ST标记和上市时间不满一年的数据行标记为1
    '''
    x['list_date'] = pd.to_datetime(x.listdate)
    x['year_1'] = np.where(x.trd_dt > x.list_date + timedelta(days=365), 0, 1)
    x.type_st = x.type_st.fillna(0)
    return x

def outlier(x, k=4.5):
    '''
    Parameters
    ==========
    x:
        原始因子值
    k = 3 * (1 / stats.norm.isf(0.75))
    '''
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    uplimit = med + k * mad
    lwlimit = med - k * mad
    y = np.where(x >= uplimit, uplimit, np.where(x <= lwlimit, lwlimit, x))
    return pd.DataFrame(y, index=x.index)

def z_score(x):
    return (x - np.mean(x)) / np.std(x)

def neutralize(x, factor, ind, cap='ln_cap'):
    '''
    Parameters
    ===========
    x:
        包含标准化后的因子值的DataFrame
    ind: str
        排除第一行业代码后的m-1个行业代码
    
    Returns
    =======
    res:
        标准化因子对行业哑变量矩阵和对数市值回归后的残差
    '''
    a = np.array(x.loc[:, ind + [cap]])
    A = np.hstack([a, np.ones([len(a), 1])])
    y = x.loc[:, factor]
    beta = np.linalg.lstsq(A, y)[0]
    res = y - np.dot(A, beta)
    return pd.DataFrame(res, index=x.index)
#    return pd.DataFrame(beta, index=ind +[cap])
#    params = bpd.groupby('trd_m').apply(neutralize, ind).unstack()

#def retn(x):
#    r = x.groupby('stkcd').apply(lambda x: x.pct_change())
#    return r

def clean(x, f, lv=1, indcd='wind_indcd'):
    '''
    Parameters
    ==========
    x: DataFrame
        含有因子原始值、市值、行业代码
    f:
        因子名称
    '''
#    x[f + '_out'] = x[f + '_raw'].groupby(level=lv).apply(outlier)
    x[f + '_out'] = x[f].groupby(level=lv).apply(outlier)
    x[f + '_zsc'] = x[f + '_out'].groupby(level=lv).apply(z_score)
#    x[f + '_zsc'] = x[f].groupby(level=lv).apply(z_score)
    x['wind_2'] = x[indcd].apply(str).str.slice(0, 6)
    x = x.join(pd.get_dummies(x['wind_2'], drop_first=True))
    x['ln_cap'] = np.log(x['cap'])
    ind = list(np.sort(x['wind_2'].unique()))[1:]
    x[f + '_neu'] = x.groupby(level=lv).apply(neutralize, f + '_zsc', ind)
#    x['retn'] = retn(x['adjclose'])
#    x = x.dropna()
    return x


def bench(table='Equity_selected_indice_ir', m='zz500', rf='shibor1m'):
    '''
    Parameters
    ==========
    m: 市场指数, 默认为zz500
    rf: 无风险利率，默认是Shibor1m
    
    Returns
    =======
    return mkt_idx, rf #元组的形式返回结果
    mkt_idx: 
        市场指数的月收益率
    rf:
        无风险利率的的历史平均值
   
    '''
    query = ('SELECT trd_dt, ' + m + ', ' + rf + ' FROM ' + table)
    columns = ['trd_dt', m, rf]
    market = get_data(query, columns)
    rf = market.shibor1m.mean()

    market['trd_date'] = pd.to_datetime(market.trd_dt)
    market['trd_m'] = market.trd_dt.str.slice(0, 6)
    market = market.resample('M', on='trd_date').last()
    mkt_idx = market.loc[:, ['trd_m', m]].set_index('trd_m').pct_change().dropna()
    return mkt_idx, rf

#cp_data = pd.DataFrame(close_price_post.stack(), columns=['close_price_post']).reset_index()
#cp_m = cp_data.groupby('stock_ID').resample('M', on='trade_date').last() #是否填充停牌数据
#cp_m['retn'] = cp_m['close_price_post'].groupby('stock_ID').apply(lambda x: x.pct_change())
#cp_m = cp_m.dropna(how='all')
def retn(cp_m, s=1, idx=['trade_date', 'stock_ID']):
    cp_m['retn_' + str(s) + 'm'] = cp_m.groupby('stock_ID')['retn'].shift(-s)
    cp_m = cp_m.set_index(idx, drop=False)
    r = cp_m[['retn_' + str(s) + 'm']]
    return r
    
#mk_data = pd.DataFrame(zz500, columns=['zz500']).reset_index()
#mk_m = mk_data.resample('M', on='trade_date').last()
#mk_m['retn_1m_zz500'] = mk_m['zz500'].pct_change().shift(-1)
#mk_m = mk_m.set_index('trade_date').dropna()
