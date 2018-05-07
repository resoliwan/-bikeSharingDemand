## Elaspsed data
from datetime import datetime

t1 = "2011-01-20 00:00:00"
t2 = "2011-01-20 01:00:00"
d1 = datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
d2 = datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
(d2 - d1).total_seconds()/(60 * 60)

d2.month
d2.day
d2.hour
d2.weekday()

def get_elapsed_hour(base_str, target_str):
  base = datetime.strptime(base_str.strip(), "%Y-%m-%d %H:%M:%S")
  target = datetime.strptime(target_str.strip(), "%Y-%m-%d %H:%M:%S")
  return (target - base).total_seconds()/(60 * 60)

get_elapsed_hour(t1, t2)

## Load data
from numpy import genfromtxt
data = genfromtxt('./data/h_train.csv', delimiter=',', skip_header=True)
data

import csv 
import numpy as np
file_name = './data/h_train.csv'
base_time_str = "2011-01-01 00:00:00"
data = []
with open(file_name, 'r') as f:
  csv_iter = csv.reader(f, delimiter=',')
  next(csv_iter)
  for row in csv_iter:
    target = datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M:%S")
    row[0] = get_elapsed_hour(base_time_str, row[0])
    row = [target.month, target.day, target.hour, target.weekday()] + row
    data.append(row)

data = np.asarray(data, dtype=np.float32)
np.save('./data/train', data)

def save_data_np_array(load_file_name, result_file_name):
  base_time_str = "2011-01-01 00:00:00"
  data = []
  with open(load_file_name, 'r') as f:
    csv_iter = csv.reader(f, delimiter=',')
    next(csv_iter)
    for row in csv_iter:
      target = datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M:%S")
      row[0] = get_elapsed_hour(base_time_str, row[0])
      row = [target.month, target.day, target.hour, target.weekday()] + row
      data.append(row)
  data = np.asarray(data, dtype=np.float32)
  np.save(result_file_name, data)

save_data_np_array('./data/train.csv', './data/train')
save_data_np_array('./data/test.csv', './data/test')

save_data_np_array('./data/h_train.csv', './data/h_train')
save_data_np_array('./data/h_test.csv', './data/h_test')
