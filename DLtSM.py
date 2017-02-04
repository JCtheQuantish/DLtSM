# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 04:40:18 2017

@author: Jeff
"""

import re
import time
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import tensorflow as tf
from  tensorflow.contrib.learn.python.learn.estimators.dnn  import DNNClassifier
from tensorflow.contrib.layers import real_valued_column

zscore = lambda x:(x-x.mean())/x.std()
ret = lambda x,y: log(y/x)

Instrument = ['SPY', 'IBM', 'FB', 'AMZN', 'T', 'TSLA', 'AMD', 'NVDA', 'LMT',
              'EWM', 'TVIX', 'JNUG', 'BRZU', 'DUST',]

def Data(i):
    D = pdr.DataReader(i, 'yahoo')
    print(D.head())
    return D

def make_inputs(D, i):
    Res = pd.DataFrame()
    
    Res['c2o'] = np.log(D['Open']/D['Close'])
    Res['h2o'] = np.log(D['Open']/D['High'])
    Res['l2o'] = np.log(D['Open']/D['Low'])
    Res['c2h'] = np.log(D['High']/D['Close'])
    Res['c1c0'] = (np.log(D['Close']/D['Close'].shift(1))).fillna(0)
    Res['V'] = zscore(D['Volume'])
    Res['ticker'] = i
    
    print(Res.head())
    # Res['c1c0'].cumsum().plot()
    # plt.show()
    
    return Res

Final = pd.DataFrame()
for i in Instrument:
    print(i)
    time.sleep(1)
    D = Data(i)
    Res = make_inputs(D, i)
    Final = Final.append(Res)
    # print(Final.head())
    # print(Final.shape)
    # time.sleep(2.5)

pivot_columns = Final.columns[:-1]    
P = Final.pivot_table(index=Final.index, columns='ticker', values=pivot_columns)
print(P.head())

mi = P.columns.tolist()
new_ind = pd.Index(e[1] + '_' + e[0] for e in mi)
P.columns = new_ind
P = P.sort(axis=1)
print(P.head())

clean_and_flat = P.dropna(1)

target_cols = list(filter(lambda x: 'c1c0' in x, clean_and_flat.columns.values))
input_cols = list(filter(lambda x: 'c1c0' not in x, clean_and_flat.columns.values))

InputDF = clean_and_flat[input_cols]
TargetDF = clean_and_flat[target_cols]

corrs = TargetDF.corr()

num_stocks = len(TargetDF.columns)

TotalReturn = ((1-np.exp(TargetDF)).sum(1))/num_stocks

def labeler(x):
    if x > 0.01:
        return 1
    if x < -0.01:
        return -1
    else:
        return 0
        
Labeled = pd.DataFrame()
Labeled['Return'] = TotalReturn
Labeled['class'] = TotalReturn.apply(labeler, 1)
Labeled['multi_class'] = pd.qcut(TotalReturn,11,labels=range(11))

def labeler_multi(x):
    if x > 0.01:
        return 1
    if x < -0.01:
        return -1
    else:
        return 0
        
Labeled['act_return'] = Labeled['class'] * Labeled['Return']
Labeled[['Return', 'act_return']].cumsum().plot(subplots=True)
plt.show()