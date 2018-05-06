from sklearn import linear_model
from datetime import datetime

t1 = "2011-01-20 00:00:00"
t2 = "2011-01-20 01:00:00"
d1 = datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
d2 = datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
(d2 - d1).total_seconds()/(60 * 60)

def get_elapsed_hour(base_str, target_str):
  base = datetime.strptime(base_str, "%Y-%m-%d %H:%M:%S")
  target = datetime.strptime(target_str, "%Y-%m-%d %H:%M:%S")
  return (target - base).total_seconds()/(60 * 60)

get_elapsed_hour(t1, t2)
