from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import numpy as np

train = np.load('./data/train.npy')
train_size = int(train.shape[0] * 0.8)
train_size

X_train = train[:train_size, :-3]
X_train[0]
y_train = train[:train_size, -1::1]
y_train[0]

X_test = train[train_size:, :-3]
X_test[0]

y_test = train[train_size:, -1::1]
y_test[0]

regr = MLPRegressor(solver='adam', alpha=1e-5)

regr.fit(X_train, np.ravel(y_train))
y_pred = regr.predict(X_test)

print(y_test)
y_pred[y_pred < 0] = 0
print(y_pred)

print('Mean squre error: ', mean_squared_error(y_test, y_pred))
print('Mean squre log error: ', mean_squared_log_error(y_test, y_pred))


