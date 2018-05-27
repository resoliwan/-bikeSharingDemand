from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
import numpy as np

train = np.load('./data/train.npy')
train_size = int(train.shape[0] * 1)
train_size

X_train = train[:train_size, :-3]
X_train[0]
y_train = train[:train_size, -1::1]
y_train[0]

regr = MLPRegressor(solver='adam', alpha=1e-10, hidden_layer_sizes=(100, 100, 100), verbose=True, tol=1e-20, learning_rate='adaptive', learning_rate_init=1e-5, max_iter=1000)
regr.fit(X_train, np.ravel(y_train))

test = np.load('./data/test.npy')
X_test = test

y_pred = regr.predict(X_test)
np.floor(y_pred)

import csv 
data = []
with open('./data/test.csv', 'r') as f:
  csv_iter = csv.reader(f, delimiter=',')
  next(csv_iter)
  for row in csv_iter:
    data.append(row)

y_pred = np.floor(y_pred)
data = np.asarray(data)

data[:,0].shape

y_pred.shape

result = np.stack((data[:,0], y_pred), axis=1)

np.savetxt('./data/ne1.csv', result, fmt='%s', delimiter=',', header='datetime,count')

print(y_test)
y_pred[y_pred < 0] = 0
print(y_pred)

print('Mean squre error: ', mean_squared_error(y_test, y_pred))

print('Mean absolute error: ', mean_absolute_error(y_train, y_pred))

print('Mean squre log error: ', mean_squared_log_error(y_train, y_pred))

a = np.array([1,2])
b = np.array([3,4])
np.stack((a, b), axis=1)
