#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn import preprocessing
import collections

look_back_period = 30
future_look_period = 3

data = pd.read_csv('ETH-USD.csv', names='time low high open close volume'.split())
data.set_index('time', inplace = True)

data.sort_index(inplace = True)
data = data[['close','volume']]
data['future_value'] = data['close'].shift(-future_look_period)
data.dropna(inplace = True)
data['target'] = data.apply(lambda row : 1 if (row['future_value'] >  row['close']) else 0, axis = 1) 
trainSet, testSet,_ = np.split(data,[int(0.95*len(data)),len(data)])

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
    return np.array(inputs), np.array(decision)

trainInputs, trainTargets = preprocess_data(trainSet)
testInputs, testTargets = preprocess_data(testSet)

model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid',return_sequences=True,dropout=0.2,input_shape = trainInputs.shape[1:]),
        tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid',return_sequences=True,dropout=0.2),
        tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid',return_sequences=False,dropout=0.2),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(2,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optmizer='adam',
              metrics=['accuracy']
             )

model.fit(trainInputs, trainTargets,validation_data=(testInputs, testTargets),epochs=5,batch_size=40)

model.save('cryptoPrediction.model')

