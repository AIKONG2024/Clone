import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#데이터 불러오기
path = "C:/_data/dacon/iris/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# print(train_csv.columns)

x = train_csv.drop(columns="species", axis= 1)
y = train_csv['species']

#원핫
one_hot_y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x,one_hot_y, train_size=0.7, random_state=200, stratify= one_hot_y)

#모델 구성 확인
print(train_csv.shape)#(120, 6)
print(test_csv.shape)#(30, 5)
print(submission_csv.shape)#(30, 2)

#모델 구성
model = Sequential()
model.add(Dense(64, input_dim = len(x.columns)))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(3, activation="softmax"))

#모델 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#모델 평가, 예측
loss = model.evaluate(x_test, y_test)
submission = np.argmax(model.predict(test_csv), axis=1)

print(submission)
