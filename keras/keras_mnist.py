from keras.datasets import mnist
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape , y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)
unique, count =  np.unique(y_train, return_counts=True)
print(unique, count) #[0 1 2 3 4 5 6 7 8 9] [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

unique, count =  np.unique(y_train, return_counts=True) #[0. 1.] [540000  60000]
print(unique, count) 

#scaling
x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

#Feature extraction
#Conv Layerset 1
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape = (28,28,1), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#Conv Layerset 2
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#Classification
#Flatten Layer
model.add(Flatten())
#Fully Connected Layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) #class count : 10

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=3, batch_size=2000, validation_split=0.2)
predict = ohe.inverse_transform(model.predict(x_test))
print(predict)