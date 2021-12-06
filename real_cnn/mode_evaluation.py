import yfinance
import numpy as np
import os
from keras import callbacks
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import load_model, model_from_json
## LSTM
from keras.layers import LSTM

## CNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from real_cnn.connMysql import Mysql


class Predictor():
    def __init__(self):
        super().__init__()
        self.mysql = Mysql()

        self.model = None
        self.x_real = None
        self.y_real = None
        self.x_train = None
        self.y_train = None

        path = "c:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc('font', family=font_name)

    def normalize_data(self, dataset):
        cols = dataset.columns.tolist()
        col_name = [0] * len(cols)
        for i in range(len(cols)):
            col_name[i] = i
        dataset.columns = col_name
        dtypes = dataset.dtypes.tolist()
        min_max = list()
        for column in dataset:
            dataset = dataset.astype({column: 'float32'})
        for i in range(len(cols)):
            col_values = dataset[col_name[i]]
            value_min = min(col_values)
            value_max = max(col_values)
            min_max.append([value_min, value_max])
        for column in dataset:
            values = dataset[column].values
            for i in range(len(values)):
                values[i] = (values[i] - min_max[column][0]) / (min_max[column][1] - min_max[column][0])
            dataset[column] = values
        dataset[column] = values
        return dataset, min_max

    def split_sequences(self, sequence, n_steps):
        print('split_sequences start')
        # print(type(sequence))
        x, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence) - 1:
                break

            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            print(seq_x)
            """
            [[0.88       1.         0.9005848  0.90697676]
             [0.92       0.9404762  0.88304096 0.9186047 ]
             [0.8457143  0.8452381  0.80701756 0.79651165]
             [0.82285714 0.85714287 0.83040935 0.85465115]
             [0.86285716 0.86904764 0.8596491  0.8255814 ]
             [0.8742857  0.9047619  0.84210527 0.80813956]
             [0.7942857  0.77380955 0.7777778  0.74418604]
             [0.86285716 0.8869048  0.877193   0.89534885]
             [0.9028571  0.97619045 0.9298246  0.93604654]
             [0.8685714  0.86904764 0.8596491  0.8372093 ]]
            """
            # print(seq_y)
            """
            [0.82857144 0.8333333  0.80701756 0.7732558 ]
            """
            x.append(seq_x)
            y.append(seq_y)
        print('split_sequences end')
        return np.array(x), np.array(y)

    def data_setup(self, n_steps, sequence): # n_seq,

        print('data_setup start')
        print('sequence ==== ')
        # print(sequence)
        """
        [[0.74285716 0.8214286  0.7426901  0.8255814 ]
         [0.88       1.         0.9005848  0.90697676]
         [0.92       0.9404762  0.88304096 0.9186047 ]
         ............................................
         [0.29714286 0.3690476  0.32163742 0.40697673]]
        """
        x, y = self.split_sequences(sequence, n_steps)
        n_features = x.shape[2]
        x = x.reshape((len(x), n_steps, n_features))
        new_y = []
        for term in y:
            new_term = term[-1]
            new_y.append(new_term)

        print('data_setup end')
        return x, np.array(new_y), n_features


    def loadData(self, code=None):
        getCount = 480
        period = 10
        if code:
            df = self.mysql.prices(code, getCount)
            # df = df.drop(['open'], 1).drop(['high'], 1).drop(['low'], 1)
            df = df.sort_values(by=['ymd'])
            # print(df['ymd'])
            # df['ymd'] = pd.to_datetime(df.ymd, format='%Y-%m-%d')
            # df['ymd'].dt.strftime('%Y-%m-%d')
            df.index = df['ymd']
            df.set_index('ymd', inplace=True)
        # print(df)
        """
                     open   high    low  close
        ymd                                   
        2021-02-01  81700  83400  81000  83000
        2021-02-02  84100  86400  83700  84400
        2021-02-03  84800  85400  83400  84600
        2021-02-04  83500  83800  82100  82500
        2021-02-05  83100  84000  82500  83500
        ...           ...    ...    ...    ...
        2021-11-26  73500  74100  72000  72300
        2021-11-29  71700  73000  71400  72300
        2021-11-30  73200  73900  70500  71300
        2021-12-01  72000  74800  71600  74400
        2021-12-02  73900  75800  73800  75800
        """

        dataset, min_max = self.normalize_data(df)
        # print('dataset after normalize================', dataset)
        """
        [[ 0.90122801  0.91852701  0.88839298  0.91796899]
         [ 0.93638402  1.00446403  0.907924    0.99944198]
         [ 0.96651798  0.98772299  0.90346003  0.91517901]
         ...
         [72.77999878 73.49250031 72.02999878 72.44999695]
         [72.36499786 73.17250061 71.30500031 72.87999725]
         [72.48249817 73.41999817 72.37999725 73.41249847]]
        """
        values = dataset.values # 하루가 부족함
        # print('dataset.values after normalize================', dataset.values)

        n_steps = 10
        # n_steps = 10
        # n_seq = 10000
        rel_test_len = 0.1


        x, y, n_features = self.data_setup(n_steps, values)  # n_seq,

        print('after data_seup 1: ', y)
        x = x[:-1]  # 종가를 제외한 모든 열
        y = y[1:]

        print('after data_seup 2: ', y)

        # print(len(x), int(len(x) * rel_test_len))
        # self.x_real, self.y_real = x[:int(len(x) * rel_test_len)], y[:int(len(x) * rel_test_len)]
        # self.x_train, self.y_train = x[int(len(x) * rel_test_len):], y[int(len(x) * rel_test_len):]
        # print(len(x), int(len(x) * rel_test_len))
        # test_len = int(len(x) * rel_test_len)
        # test_len = int(len(x) - rel_test_len)
        test_len = int(len(x) * rel_test_len) # 세트간의 분할은 10-90 이고 90%는 학습에 사용되고 10%는 현재 값이다
        # print(str(test_len) + '=' + str(len(x)) + '*' + str(rel_test_len))
        """ 19=197*0.1 """
        # self.x_real, self.y_real = x[:test_len], y[:test_len]
        # self.x_train, self.y_train = x[test_len:], y[test_len:]
        print('type of x : ', type(x))  # <class 'numpy.ndarray'>
        self.x_train, self.y_train = x[:-test_len], y[:-test_len]
        self.x_real, self.y_real = x[len(x) - test_len:], y[len(x) - test_len:]


        # print('self.x_train ===================')
        # print(self.x_train);
        # # print('self.x_real ===================')
        # # # print(self.x_real)
        # # print('self.y_real ===================')
        # # print(self.y_real)
        # # print(self.x_real, self.y_real)
        # print('x.shape', x.shape)
        """
        (109, 10, 4)
        """

    def cnn(self):
        self.model = Sequential()
        # self.model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(10, 4)))
        self.model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(10, 4)))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='mse', optimizer='adam')

    def train_networks(self, file):
        epochs = 5000
        verbosity = 2
        dirx = './'
        os.chdir(dirx)
        h5 = file + '.h5'
        checkpoint = callbacks.ModelCheckpoint(h5,
                                               monitor='val_loss',
                                               verbose=0,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               mode='auto',
                                               period=1)
        callback = [checkpoint]
        json = file + '.json'
        model_json = self.model.to_json()
        with open(json, "w") as json_file:
            json_file.write(model_json)

        # model.fit(x, y, batch_size=32, epochs=10)
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=len(self.x_train) // 4,
            epochs=epochs,

            validation_data=(self.x_real, self.y_real),
            verbose=verbosity,
            callbacks=callback)

    def load_keras_model(self, optimizer, file):
        dirx = './'
        os.chdir(dirx)
        json_file = open(file + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.compile(optimizer=optimizer, loss='mse')
        model.load_weights(file + '.h5')
        return model

    # def drawGraph(self, **args):
    #     """
    #     그래프 출력
    #     :return:
    #     """
    #     title = None
    #     if args['title']:
    #         title = args['title']
    #
    #     self.model.evaluate(self.x_real, self.y_real)
    #
    #     pred_test = self.model.predict(self.x_real)
    #     plt.plot(pred_test, 'r')
    #     plt.plot(self.y_real, 'g')
    #     plt.title(title)
    #     plt.legend(['예측값', '실제값'])
    #     plt.show()

    def drawGraph(self, **args):
        """
        그래프 출력
        :return:
        """
        title = None
        if args['title']:
            title = args['title']

        self.model.evaluate(self.x_train, self.y_train)

        pred_test = self.model.predict(self.x_train)
        plt.plot(pred_test, 'r')
        plt.plot(self.y_real, 'g')
        plt.title(title)
        plt.legend(['예측값', '실제값'])
        plt.show()

if __name__ == "__main__":
    pred = Predictor()
    pred.loadData('005930')
    pred.cnn()
    # pred.train_networks('005930')
    pred.model = pred.load_keras_model('adam', '005930')
    pred.drawGraph(title='CNN에 의한 예측결과 ')

    # pred.byCnn('005930')  # 삼성전자 (005930)
    # pred.byProphet('253450')
    # pred.byProphet('011210')
    # pred.byProphet('000660') # sk 하이닉스
    # pred.byProphet('272210')
    # pred.byProphet('278280') # 천보
