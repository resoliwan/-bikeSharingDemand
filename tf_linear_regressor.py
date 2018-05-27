import math
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

pd.options.display.max_rows = 10

df = pd.read_csv('./data/train.csv')

# df.describe()
df.columns
# df[0:2]

np.array([['a'], ['b']]).shape
np.array(['a']).shape

# features = df[['weather']] 263
# 263
# columns = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
# columns = ['season']
columns = ['temp']
plt.show()
features = df[columns]

# features.shape

feature_columns = [tf.feature_column.numeric_column(column) for column in columns];
targets = df['count']

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=optimizer
    )

def input_fn(features, targets, batch_size=1, shuffle=True, num_eponchs=None):
  features = {key: np.array(value) for key, value in features.items()};
  print('features', features)
  ds = Dataset.from_tensor_slices((features, targets))
  ds = ds.batch(batch_size).repeat(num_eponchs)
  if(shuffle):
    ds = ds.shuffle(buffer_size=10000)
  features, labels = ds.make_one_shot_iterator().get_next()
  return features, labels

input_fn(features, targets)

_ = linear_regressor.train(input_fn=lambda:input_fn(features, targets), steps=1000)

prediction_input_fn = lambda: input_fn(features, targets, num_eponchs=1, shuffle=False)
predictions = linear_regressor.predict(input_fn=prediction_input_fn)
predictions = np.array([item['predictions'][0] for item in predictions])
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print('Mean Squared Error(on training data): %0.3f' % mean_squared_error)
print('Root Mean Squared Error (on trining data): %0.3f' % root_mean_squared_error)

data = pd.DataFrame()
data['predictions'] = pd.Series(predictions)
data['targets'] = pd.Series(targets)
data.describe()

sample = df.sample(n=300)

from matplotlib import pyplot as plt

x_0 = sample['temp'].min()
x_1 = sample['temp'].max()

linear_regressor.get_variable_names()
weight = linear_regressor.get_variable_value('linear/linear_model/temp/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

y_0 = x_0 * weight + bias
y_1 = x_1 * weight + bias

plt.plot([x_0, x_1], [y_0, y_1], c='r')

plt.scatter(sample['temp'], sample['count'])
plt.show()






