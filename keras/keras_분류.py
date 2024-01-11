
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

#One Hot Encoder
# ohe_y =  pd.get_dummies(y)

# from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-1, 1)
# print(y)
# ohe_y = OneHotEncoder().fit_transform(y)

ohe_y = to_categorical(y)
ohe_y = np.delete(ohe_y,0,axis=1)
print(ohe_y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, ohe_y, train_size=0.8, random_state=2200, stratify=ohe_y)


model = Sequential()
model.add(Dense(64, input_dim = 54))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(7, activation='softmax'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=1, batch_size=10000, verbose= 1, validation_split=0.3, callbacks=[es])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

arg_y_test = np.argmax(y_test, axis=1)
arg_y_predict = np.argmax(y_predict, axis=1)

from sklearn.metrics import accuracy_score
accu_score = accuracy_score(arg_y_test, arg_y_predict)

print(accu_score)


