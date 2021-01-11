import sys
sys.path.append("/home/ross/quantfin/lib")

import math
import time
from datetime import datetime
import importlib
import numpy as np # numerical computation packages in python
import matplotlib.pyplot as plt # plotting routines
import pandas as pd
import multiprocessing as mp
import AFMLlib as afml
importlib.reload(afml) # Reload module in case it changed in interactive mode

from statsmodels.tsa.stattools import adfuller
from joblib import dump, load
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from joblib import dump, load

colNames = ["side", "readtime", "filltime", "volume", "price"]
rawTradesData = pd.read_csv('data/XXXX.csv', sep=';', index_col=1, names=colNames, parse_dates=[1, 2], nrows=7000000)

tradesData = rawTradesData
# tradesData = rawTradesData.head(math.floor(rawTradesData.shape[0] * 0.3)).sort_index()

clf = load('models/VolumeML.joblib')

volCandles = afml.createVolumeCandles(tradesData, frequency=5)
closePrices = volCandles.Close
fracDiffClose = afml.frac_diff_ffd(closePrices, 0.35) # Fractionally differenced close prices
fracDiffKurtosis = fracDiffClose.rolling(400).kurt()
cusumEvents = afml.getCUSUMEvents(fracDiffClose, 5) # Timestamps of CUSUM events

barrierWidth = 1 # Width of the top and bottom barriers
volCandles['targets'] = 0.05
targets = volCandles['targets'] # Target returns for each event
minReturn = 0.01 # Minimum target return to run a triple barrier search
numThreads = 4 # Number of processor threads to use

# Series of timestamps for the third 'timeout' barriers
timeBarriers = closePrices.index.searchsorted(cusumEvents+pd.Timedelta(hours = 4)) # 4 hours in the future
timeBarriers = timeBarriers[timeBarriers<closePrices.shape[0]]
timeBarriers = pd.Series(closePrices.index[timeBarriers],index = cusumEvents[:timeBarriers.shape[0]]) # NaNs at end

# Calculate returns for each event
events = afml.getEvents(closePrices, cusumEvents, barrierWidth, targets, minReturn, numThreads, timeBarriers)

bins = afml.getBins(events, closePrices)
bins.bin[bins.ret < minReturn] = 0

# Split data into train and test
X = pd.concat([fracDiffClose, fracDiffKurtosis], axis=1)
X = X.loc[bins.index]
X = X[~np.isnan(X).any(axis=1)]
y = bins.loc[X.index].bin

y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred)
score = cm[1,1] / (cm[0,1] + cm[1,1])
print(cm)
print(score)
