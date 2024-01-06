

# 구상
# #0. Data 전처리
# 1-1 데이터 분석
'''
id 고유 id
hour 시간
temperature 기온
precipitation 비가 오지 않았으면 0, 비가 오면 1
windspeed 풍속(평균)
humidity 습도
visibility 시정(視程), 시계(視界)(특정 기상 상태에 따른 가시성을 의미)
ozone 오존
pm10 미세먼지(머리카락 굵기의 1/5에서 1/7 크기의 미세먼지)
pm2.5 미세먼지(머리카락 굵기의 1/20에서 1/30 크기의 미세먼지)
count 시간에 따른 따릉이 대여 수

train_csv null 값

id                          0
hour                        0
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
count                       0
dtype: int64

test_csv null 값

id                         0
hour                       0
hour_bef_temperature       1
hour_bef_precipitation     1
hour_bef_windspeed         1
hour_bef_humidity          1
hour_bef_visibility        1
hour_bef_ozone            35
hour_bef_pm10             37
hour_bef_pm2.5            36
dtype: int64

결측치가 있는 부분
temperature
precipitation 
windspeed 
humidity 
visibility 
ozone 
pm10 
pm2.5 


각 컬럼의 결측치를 어떻게 처리할 것인가?
내가 아는 방식 : 0 혹은 평균.

temperature : 평균
precipitation : 0
windspeed :평균
humidity :평균 
visibility :평균
ozone : 평균
pm10 :평균
pm2.5 :평균



'''



# 1. 데이터 불러오기
import pandas as pd
path = '/Users/kongseon-eui/Documents/Workspace/AI_Project/_data/'
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'submission.csv')

# 2. 데이터 정규화
#데이터 모양
# print(train_csv.shape)  # (1459, 11)
# print(test_csv.shape)  # (715, 10)
# print(submission_csv.shape)  # (715, 2)

#결측치 제거

#train
train_csv['hour_bef_precipitation'].fillna(value= 0.0, inplace=True)
# train_csv.fillna(test_csv.mean(), inplace= True)
train_csv.fillna(train_csv.interpolate(method='values'), inplace=True)

#test
test_csv['hour_bef_precipitation'].fillna(value= 0.0 , inplace=True)
# test_csv.fillna(test_csv.mean(), inplace=True)
test_csv.fillna(test_csv.interpolate(method='values'), inplace=True)




# print(train_csv.isna().sum())
# print(test_csv.isna().sum())

#데이터 x,y 자름
x = train_csv.drop('count', axis= 1)
# x = train_csv.drop(columns='count')
y = train_csv['count']

# print(x.shape)#(1459, 10)
# print(y.shape)#(1459,)

# best_random_state = 0
# import random
# random_random_state = random.randint(0,4290000000)
# 3. 데이터 셔플 및 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.75, random_state=6131483)

##################################################데이터 전처리

from keras.models import Sequential
from keras.layers import Dense

# 모델구성
model = Sequential()
model.add(Dense(32, input_dim = 10))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=20) #epo : 10, 50, 100  /// bat : 10, 20, 40, 60 ,80

#평가 ,예측
loss = model.evaluate(x_test, y_test)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
submit = model.predict([test_csv])
print(submit.shape)
print(y_predict.shape)
# r2_2 = r2_score(y_test, submit[:365])/

print('loss : ', loss)
print('r2 : ', r2)
# print('r2_2 : ', r2_2)

#sumission.csv 파일 생성
submission_csv['count'] = submit

import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}.csv", index=False)

#그래프 확인
import matplotlib.pyplot as plt
plt.scatter(y_test, y_predict)
# plt.plot(y_test, color = 'red')  # Plotting actual values
plt.show()
