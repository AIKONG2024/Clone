from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import numpy as np
import pandas as pd

#이미지 가져오기
np_path = '../_data/_save_npy/'
xy_train = np.load(np_path + 'keras_clone_brain_xy_train.npy', allow_pickle=True)
xy_test = np.load(np_path + 'keras_clone_brain_xy_test.npy', allow_pickle=True)

print(xy_train[0][0])
print(xy_train[0][1])
print(xy_test[0][0])
print(xy_test[0][1])