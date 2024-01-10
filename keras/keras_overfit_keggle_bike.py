
#데이터 가져오기

path = 'C:/_data/kaggle/bike/'
import pandas as pd
train_csv = pd.read_csv(path + "train.csv", index_col='datetime') 
test_csv = pd.read_csv(path + "test.csv", index_col='datetime')
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

#데이터 확인

# print(train_csv.shape)
# print(test_csv.shape)
# print(submission_csv.shape)

# print(train_csv.info)
# print(test_csv.info)

#데이터 전처리
x = train_csv.drop(columns='count')
y = train_csv['count']

x.drop(columns='casual', inplace=True)
x.drop(columns='registered', inplace=True)

#데이터 구조 및 결측치 확인 / 제거
# print(x.columns)
# print(x.shape)
# print(y.shape)
# print(x.isna().sum()) #(10886, 8)
# print(y.isna().sum()) #(10886,)

#훈련, 평가 데이터 쪼개기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1234)
print(x_train.shape)
print(y_train.shape)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(512, input_dim = len(x.columns)))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(4, activation= 'relu'))
model.add(Dense(2, activation= 'relu'))
model.add(Dense(1))

#모델 컴파일, 훈련
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience= 20, verbose= 1)
model.compile(loss='mse', optimizer='adam')
history = model.fit(x_train, y_train, epochs= 600, batch_size=20, validation_split=0.3, verbose=1, callbacks=[es])

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
#평가, 예측
y_predict = model.predict(x_test)

rmse_loss = np.sqrt(mean_squared_error(y_test, y_predict))
rmsle_loss = np.sqrt(mean_squared_log_error(y_test, y_predict))
r2_score = r2_score(y_test, y_predict)

print('rmse_loss :', rmse_loss)
print('rmsle_loss :', rmsle_loss)
print('r2_score :', r2_score)

#예측값
submission = model.predict(test_csv)

#파일 출력
submission_csv.to_csv(path + "sampleSubmission_0110.csv", index=False)

#히스토리
losses = history.history['loss']
val_losses = history.history['val_loss']

#그래프 출력
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(losses, c = 'red', label = 'loss', marker = '.')
plt.plot(val_losses, c = 'blue', label = 'val_loss', marker = '.')
plt.legend(loc = 'upper right')
plt.title("바이크 찻트")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.show()


