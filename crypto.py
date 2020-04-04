#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn import preprocessing
import collections


# In[2]:


look_back_period = 30
future_look_period = 3


# In[3]:


data = pd.read_csv('ETH-USD.csv', names='time low high open close volume'.split())


# In[4]:


data.set_index('time', inplace = True)


# In[5]:


data.sort_index(inplace = True)


# In[6]:


data = data[['close','volume']] 


# In[7]:


data['future_value'] = data['close'].shift(-future_look_period)
data.dropna(inplace = True)


# In[8]:


data['target'] = data.apply(lambda row : 1 if (row['future_value'] >  row['close']) else 0, axis = 1) 


# In[9]:


trainSet, testSet,_ = np.split(data,[int(0.95*len(data)),len(data)])


# In[10]:


def preprocess_data(data_pre):
    data_pre.drop('future_value',axis = 1)
    for col in data_pre.columns:
        if col != 'target':
            data_pre[col] = data_pre[col].pct_change()
            data_pre.dropna(inplace = True)
            data_pre[col] = preprocessing.scale(data_pre[col].values)
    data_pre.dropna(inplace = True)
    sequential_data = []
    prev_days = collections.deque(maxlen = look_back_period)
    for i in data_pre.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == look_back_period:
            sequential_data.append([np.array(prev_days),i[-1]])
    random.shuffle(sequential_data)
    buys = []
    sells= []
    for seq,target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        else:
            buys.append([seq,target])
    low_value = min(len(sells),len(buys))
    buys = buys [:low_value]
    sells = sells [:low_value]
    final_sequence = buys+sells
    random.shuffle(final_sequence)
    inputs = []
    decision = []
    for seq, target in final_sequence:
        inputs.append(seq)
        decision.append(target)
    return np.array(inputs), decision


# In[11]:


a, b = preprocess_data(trainSet)
a.shape


# In[12]:


a, b = preprocess_data(testSet)
a.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().system('jupyter nbconvert --to script crypto.ipynb')

