from keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_generator = ImageDataGenerator(
    rescale=1./255, #부동소수점 연산 : 연산속도 느림, 범위 : 넓음, 수학계산 이용, 근삿값표현
    fill_mode='nearest'
)

test_generator = ImageDataGenerator(
    rescale=1./255
)

path_train = "c:/_data/image/brain/train/" #path 는 밖으로 빼서 사용 가능하게 해줌.
path_test = "c:/_data/image/brain/test/"

#이미지 가져오기
train = train_generator.flow_from_directory(
    directory= path_train,
    batch_size=100,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    target_size=(150,150)
)

test = test_generator.flow_from_directory(
    directory= path_test,
    batch_size=100,
    class_mode='binary',
    color_mode='grayscale',
    target_size=(150,150)
)
x_train = []
y_train = [] 
x_test = []
y_test = [] 

for i, (x_batch, y_batch) in enumerate(train):
    x_train.append(x_batch)
    y_train.append(y_batch)
    print("train :", i)

for i, (x_batch, y_batch) in enumerate(test):
    x_test.append(x_batch)
    y_test.append(y_batch)
    print("test :", i)




#이미지 npy 저장
# np_path = '../_data/_save_npy/'
# np.save(np_path + 'keras_clone_brain_x_train.npy', arr= x_train)
# np.save(np_path + 'keras_clone_brain_y_train.npy', arr= y_train)
# np.save(np_path + 'keras_clone_brain_x_test.npy', arr= x_test)
# np.save(np_path + 'keras_clone_brain_y_test.npy', arr= y_test)
