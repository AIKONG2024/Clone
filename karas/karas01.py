import tensorflow as tf
print(tf.__version__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 모델생성
model = Sequential()
model.add(Dense(1, input_dim = 1)) #가중치 초기화방식 : 균일분포, 활성화람수: recified linear unit
model.add(Dense(10000))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#평가, 예측
print("평가: ", model.evaluate(x,y))
print("예측: ", model.predict([4]))