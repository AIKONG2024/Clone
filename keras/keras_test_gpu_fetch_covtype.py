from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(pd.value_counts(y))

# scikit learn
y = y.reshape(-1,1)
ohe_y = OneHotEncoder(sparse=False).fit_transform(y)
# print(ohe_y.shape)#(581012, 7)
print(np.unique(ohe_y, return_counts=True) )
pd_y = pd.DataFrame(ohe_y)
# print(pd_y.columns)

#keras
ohe_y = to_categorical(y)
ohe_y = ohe_y[:,1:]
print(ohe_y.shape)
print(np.unique(ohe_y, return_counts=True) )

x_train, x_test, y_train, y_test = train_test_split(x, ohe_y, train_size= 0.7, random_state=1234, stratify=ohe_y)
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구성
model = Sequential()
model.add(Dense(64, input_shape=(54,)))
model.add(Dense(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

#컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
history = model.fit(x_train, y_train, epochs=1000, batch_size=1000, validation_split=0.2)
end_time = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
arg_y_predict = np.argmax(y_predict, axis=1)
arg_y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(arg_y_predict,arg_y_test)

print("loss : ", loss)
print("acc_score : ", acc_score)
#걸린시간 측정 CPU GPU 비교
print("걸린시간 : ", round(end_time - start_time, 2), "초")


# 걸린시간 :  439.79 초