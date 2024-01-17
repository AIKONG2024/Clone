from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


#데이터 가져오기
path = '../_data/dacon/dechul/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

#데이터 전처리
#수치화 : 대출기간, 근로기간, 주택소유상태, 대출목적, 대출등급
train_le = LabelEncoder()
train_csv['대출기간'] = train_le.fit_transform(train_csv['대출기간'])
train_csv['근로기간'] = train_le.fit_transform(train_csv['대출기간'])
train_csv['주택소유상태'] = train_le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = train_le.fit_transform(train_csv['대출목적'])

y_le = LabelEncoder()
y_le.fit(train_csv['대출등급'])
train_csv['대출등급'] = y_le.transform(train_csv['대출등급'])

test_le = LabelEncoder()
test_csv['대출기간'] = test_le.fit_transform(test_csv['대출기간'])
test_csv['근로기간'] = test_le.fit_transform(test_csv['근로기간'])
test_csv['주택소유상태'] = test_le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = test_le.fit_transform(test_csv['대출목적'])

x = train_csv.drop(['대출등급'], axis=1) 
y = train_csv['대출등급']

one_hot_y = OneHotEncoder(sparse=False).fit_transform(y.values.reshape(-1,1))
unique, count = np.unique(one_hot_y, return_counts=True)
print(unique, count)
print(x.shape,  one_hot_y.shape) #(96294, 13) / (96294, 7)

x_train, x_test, y_train, y_test = train_test_split(x,one_hot_y, train_size=0.85, random_state=1234567)

#스케일링: 데이터 Deivision 후 독립변수 진행, 분류 진행 StandardScler 적합하다고 판단 적용.
st_scler = StandardScaler()
x_train = st_scler.fit_transform(x_train)
x_test = st_scler.fit_transform(x_test)
test_csv = st_scler.fit_transform(test_csv)

#모델구성
model = Sequential()
model.add(Dense(16, input_shape=(13,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))
import datetime as dt
strftime = dt.datetime.now().strftime("%m%d_%H%H")
mcp_save_path = '../data/_save/MCP/dacon/dechul/' 
# '{epoch:04d}-{val_loss:.4f}.hdf5'
log = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp_file_path = "".join([mcp_save_path + "_" + "mcp_dechul_" + strftime + '_' + log])
print(mcp_file_path)

#모델 훈련, 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, verbose='auto', batch_size=10000, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode = 'min', patience=1000, restore_best_weights=True),
    ModelCheckpoint(filepath = mcp_file_path, monitor= "val_loss", mode='min', save_best_only=True)
])

# model.save("save_" + mcp_file_path)

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_test_arg = np.argmax(y_test, axis=1)
f1_score = f1_score(y_test_arg, y_predict, average='macro')
print("f1_score : ", f1_score)

#Label Decoding
submission = np.argmax(model.predict(test_csv), axis=1)
submission = y_le.inverse_transform(submission)

import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv['대출등급'] = submission
file_path = path + f"sampleSubmission{save_time}.csv"
submission_csv.to_csv(file_path, index=False)


