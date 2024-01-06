

# 구상
# #0. Data 전처리
# 1-1 데이터 분석
'''
id 고유 id
hour 시간
temperature 기온
precipitation 비가 오지 않았으면 0, 비가 오면 1
windspeed 풍속(평균)
humidity 습도
visibility 시정(視程), 시계(視界)(특정 기상 상태에 따른 가시성을 의미)
ozone 오존
pm10 미세먼지(머리카락 굵기의 1/5에서 1/7 크기의 미세먼지)
pm2.5 미세먼지(머리카락 굵기의 1/20에서 1/30 크기의 미세먼지)
count 시간에 따른 따릉이 대여 수

train_csv null 값

id                          0
hour                        0
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
count                       0
dtype: int64

test_csv null 값

id                         0
hour                       0
hour_bef_temperature       1
hour_bef_precipitation     1
hour_bef_windspeed         1
hour_bef_humidity          1
hour_bef_visibility        1
hour_bef_ozone            35
hour_bef_pm10             37
hour_bef_pm2.5            36
dtype: int64

결측치가 있는 부분
temperature
precipitation 
windspeed 
humidity 
visibility 
ozone 
pm10 
pm2.5 


각 컬럼의 결측치를 어떻게 처리할 것인가?
내가 아는 방식 : 0 혹은 평균.

temperature : 평균
precipitation : 0
windspeed :평균
humidity :평균 
visibility :평균
ozone : 평균
pm10 :평균
pm2.5 :평균



'''

import pandas as pd
path = '/Users/kongseon-eui/Documents/Workspace/AI_Project/_data/'
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'submission.csv')

class Dacon_ddarung:
    # 1. 데이터 불러오기

    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.metrics import r2_score
    import random
    
    #save submission file
    def save_submission(file):
        import time as tm
        ltm = tm.localtime(tm.time())
        save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
        file.to_csv(path + f"submission_{save_time}.csv", index=False)
        
    #그래프 확인 
    def show_graph(realData,predictData):
        import matplotlib.pyplot as plt
        plt.scatter(realData, predictData)
        # plt.plot(y_test, color = 'red')  # Plotting actual values
        plt.show()
        
    def save_file(value):
        df = pd.DataFrame({'random_state' : [value[0]], 'epoch' : [value[1]], 'train_size' : [value[2]], 'batch_size' : [value[3]],  'gap' : [value[4]], 'loss' : [value[5]]}) 
        # dfs = pd.DataFrame(data)
        df.to_csv(path + "random_state.csv", mode='a', index=True)
        
        
    # 2. 데이터 정규화
    #데이터 모양
    # print(train_csv.shape)  # (1459, 11)
    # print(test_csv.shape)  # (715, 10)
    # print(submission_csv.shape)  # (715, 2)

    #결측치 제거

    #train
    train_csv['hour_bef_precipitation'].fillna(value= train_csv['hour_bef_precipitation'].mean() , inplace=True)
    # train_csv.fillna(test_csv.mean(), inplace= True)
    train_csv.fillna(train_csv.interpolate(method='values'), inplace=True)

    #test
    test_csv['hour_bef_precipitation'].fillna(value= train_csv['hour_bef_precipitation'].mean() , inplace=True)
    # test_csv.fillna(test_csv.mean(), inplace=True)
    test_csv.fillna(test_csv.interpolate(method='values'), inplace=True)
    

    # print(train_csv.isna().sum())
    # print(test_csv.isna().sum())

    #데이터 x,y 자름
    x = train_csv.drop('count', axis= 1)
    # x = train_csv.drop(columns='count')
    y = train_csv['count']
    
    
       #영향이 없는 컬럼 제거
    x = x.drop('hour_bef_visibility', axis=1)
    # x = x.drop('hour_bef_ozone', axis=1)
    # x = x.drop('hour_bef_precipitation', axis=1)
    # x = x.drop('hour_bef_windspeed', axis=1)
    x = x.drop('hour_bef_pm10', axis=1)
    x = x.drop('hour_bef_pm2.5', axis=1)
    
    test_t = test_csv
    test_t = test_t.drop('hour_bef_visibility', axis=1)
    # test_t = test_t.drop('hour_bef_ozone', axis=1)
    # test_t = test_t.drop('hour_bef_precipitation', axis=1)
    # test_t = test_t.drop('hour_bef_windspeed', axis=1)
    test_t = test_t.drop('hour_bef_pm10', axis=1)
    test_t = test_t.drop('hour_bef_pm2.5', axis=1)
    

    # print(x.shape)#(1459, 10)
    # print(y.shape)#(1459,)

    # best_random_state = 0
    # import random
    # random_random_state = random.randint(0,4290000000)
    # 3. 데이터 셔플 및 분리

    
    random_random_state = 0
    random_epoch = 0
    random_batch_size = 0
    random_train_size = 0.0
    while(1) :
        random_random_state = random.randint(0,10000000)
        random_epoch = random.randint(50,500)
        random_batch_size = random.randint(10,100)
        random_train_size = round(random.uniform(0.6,0.9),2)
        
        x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=random_train_size, random_state=random_random_state)

        ##################################################데이터 전처리
        
        # 모델구성
        model = Sequential()
        model.add(Dense(2, input_dim = len(x.columns)))
        model.add(Dense(3))
        model.add(Dense(4))
        model.add(Dense(3))
        model.add(Dense(2))
        model.add(Dense(1))
        

        #컴파일, 훈련
        model.compile(loss='mse', optimizer='adam')
        model.fit(x_train, y_train, epochs=random_epoch, batch_size=random_batch_size) 
        
        #평가 ,예측
        loss = model.evaluate(x_test, y_test)

        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        
        submit = model.predict([test_t])
        
        print('loss : ', loss)
        print('r2 : ', r2)   
        gap_of_loss = abs(model.evaluate(x_train,y_train) - loss)
        print('로스 차이:', gap_of_loss)
        
        #sumission.csv 파일 생성
        if loss < 2700 and gap_of_loss < 100:
            submission_csv['count'] = submit
            save_submission(submission_csv)
            save_file([random_random_state, random_epoch, random_train_size, random_batch_size, gap_of_loss, loss])
        
        
    # show_graph(y_test, y_predict)
    
    #40

