import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#데이터 가져오기
path = 'C:/_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submossion_csv = pd.read_csv(path + 'sample_submission.csv')

#데이터 전처리
x = train_csv.drop(columns='Outcome')
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=200)

#모델 구성
model = Sequential()
model.add(Dense(64, input_dim = len(x.columns)))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid')) #sigmoid #sigmoid #sigmoid #sigmoid #sigmoid #sigmoid #sigmoid #sigmoid

#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc']) #binary crossentopy #binary crossentopy #binary crossentopy #binary crossentopy
#sigmoid #sigmoid 
history = model.fit(x_train, y_train, epochs= 100, batch_size= 500, validation_split = 0.3, verbose=1, 
          callbacks=[EarlyStopping(monitor='val_acc', mode='max', patience = 100, restore_best_weights = True )])

#평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = np.round(model.predict(x_test))
submission = np.round(model.predict(test_csv))
accuracy_score = accuracy_score(y_test, np.round(y_predict))

print("loss : ", loss)
print("accuary : ", accuracy_score)

#파일 생성
submossion_csv['Outcome'] = submission #반드시 round 해줄것.
submossion_csv.to_csv(path + "submission_0110.csv",index=False)

hist_loss = history.history['loss']
hist_val_loss = history.history['val_loss']
hist_val_acc = history.history['val_acc']
hist_acc = history.history['acc']



#시각화
plt.figure(figsize=(9,6))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.plot(hist_loss, color = 'red', label = "loss", marker = '.')
plt.plot(hist_val_loss, color = 'blue', label = "val loss", marker = '.')
plt.plot(hist_acc, color = 'green', label = "acc", marker = '.')
plt.plot(hist_val_acc, color = 'brown', label = "val_acc", marker = '.')
plt.title("찻트입니다")
plt.grid()
plt.show()
