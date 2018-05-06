# Bike Sharing Demand
[host](https://www.kaggle.com/c/bike-sharing-demand/data)

# Goals
* count 를 \\(\hat y \\) 라고 할 때 
\\[ \hat y(\vec w, \vec x) = w_1x_1 + w_2x_2 + ... + w_px_p \\]

* 전체 트레이닝 데이터 \\(X \cdot \vec w - \vec y \\) 를 최소화 하는 값을 구하자.
\\[ min \lVert Xw - y\rVert_2^2 \\]

# How to 
## data transformation
### time change
* 시간을 특정 시간에서 부터 지나간 시간으로 숫자로 변경하자
- 01-01 01 = 01
- 01-01 24 = 24
- 01-02 01 = 25

### code

```python
t1 = "2011-01-20 00:00:00"
t2 = "2011-01-20 01:00:00"
d1 = datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
d2 = datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
(d2 - d1).total_seconds()/(60 * 60)
#1.0

def get_elapsed_hour(base_str, target_str):
  base = datetime.strptime(base_str, "%Y-%m-%d %H:%M:%S")
  target = datetime.strptime(target_str, "%Y-%m-%d %H:%M:%S")
  return (target - base).total_seconds()/(60 * 60)

get_elapsed_hour(t1, t2)
```

# 순서
* 특별한 튜닝을 하지 않고여러 알고리즘만 변경해 가면서 구해보자.
* 알고리즘을 정리하고 
* 에러를 기록하고
* 앙상블을 한번 써보자.

1. Ordinary Least Squares 로 베이스를 구한다.
2. log Least Squares 로 차이점을 구한다.





# Data Fields
- datetime - hourly date + timestamp  
- season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
- holiday - whether the day is considered a holiday
- workingday - whether the day is neither a weekend nor holiday
- weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
- temp - temperature in Celsius
- atemp - "feels like" temperature in Celsius
- humidity - relative humidity
- windspeed - wind speed
- casual - number of non-registered user rentals initiated
- registered - number of registered user rentals initiated
- count - number of total rentals



