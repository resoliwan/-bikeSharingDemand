from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import numpy as np

import matplotlib.pyplot as plt

train = np.load('./data/train.npy')
# np.random.shuffle(train)

train_size = int(train.shape[0] * 0.8)
train_size

X_train = train[:train_size, :-3]
X_train[0:2]

y_train = train[:train_size, -1::1]
y_train[0:2]

X_test = train[train_size:, :-3]
X_test[0:2]
y_test = train[train_size:, -1::1]
y_test[0:2]

X_train.shape[0] + X_test.shape[0]

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_train)

print(y_train)
y_pred[y_pred < 0] = 0
print(y_pred)

y_test.shape

a = np.array([[1,2]])
a.shape

np.concatenate([[1,2]], [[100,200]])

plt.scatter(y_train, y_pred, color='black')

plt.show(block=False)

print('Coefficients: ', regr.coef_)
print('Mean squre error: ', mean_squared_error(y_test, y_pred))
print('Mean squre log error: ', mean_squared_log_error(y_test, y_pred))
