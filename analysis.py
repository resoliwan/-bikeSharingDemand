import numpy as np
from numpy import genfromtxt

data = genfromtxt('./data/h_train.csv', delimiter=',', skip_header=True)
data

import pandas as pd

df = pd.read_csv('./data/train.csv')

df['time'] = df['datetime'].apply(lambda time: pd.to_datetime(time).hour)
# df['count'].sort_values(ascending=False)[0: 10]



df['count'].mean()
pd.plotting.scatter_matrix

pd.DataFrame.plot

df.plot.scatter('time', 'count')

import matplotlib.pyplot as plt

df.plot.scatter('time', 'count')

df['count'].hist(bins=50)

df['count'].mean()

plt.show()

