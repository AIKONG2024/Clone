
#1. 데이터 전처리
import pandas as pd

#csv 가져오기
path = 'C:/_data/kaggle/bike/'
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv")

#데이터 구성 확인
print(train_csv.shape)#(10886, 12)
print(test_csv.shape)#(6493, 9)
print(sampleSubmission_csv.shape)#(6493, 2)

print(train_csv.columns)
'''
Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')
'''
print(test_csv.columns)
'''
Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed'],
      dtype='object')
'''
print(sampleSubmission_csv.columns)
'''
Index(['datetime', 'count'], dtype='object')
'''

print(train_csv.head(15))
print(test_csv.head(15))
print(sampleSubmission_csv.head(15))

#결측치 확인
print("결측치")
print(train_csv.isna().sum())#결측치 없음
print(test_csv.isna().sum())#결측치 없음


#만약 결측치가 있다면
train_csv.fillna(0) #0
train_csv.fillna(train_csv.mean()) #평균

#train_csv  / casual  registered 제거
x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
print(x)
#count 분리
y = train_csv['count']

#데이터 구조 재확인
print(x.shape)#(10886, 9)
print(y.shape)#(10886,)

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,shuffle=True, train_size= 0.8, random_state= 200)
x_test_v, x_val, y_test_v, y_val = train_test_split(x_test, y_test, train_size=0.5, random_state= 200) 


#모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(32, input_dim = len(x.columns), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

#컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=10, batch_size=10, verbose= 2)
#varbose = 0 : 출력 없음, varbose =1 : 디폴트값, varbose = 2 : loss만 출력

#평가, 예측
mse_loss = model.evaluate(x_test_v, y_test_v)
submission = model.predict(test_csv)
y_predict = model.predict(x_test)

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
r2 = r2_score(y_test, y_predict)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse_loss = RMSE(y_test, y_predict)
def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
rmsle_loss = RMSLE(y_test, y_predict)

print("mse: ", mse_loss)
print("rmse: ", rmse_loss)
print("rmsle: ", rmsle_loss)
print("r2 :", r2)

#sampleSubmission_csv음수 확인
print("음수 확인 :" ,sampleSubmission_csv[sampleSubmission_csv['count'] < 0].count())


#내보내기
sampleSubmission_csv['count'] = submission
sampleSubmission_csv.to_csv(path + "sampleSubmission_0108.csv", index=False)
