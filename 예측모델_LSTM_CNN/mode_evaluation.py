## https://medium.com/hands-on-data-science/lstms-or-cnns-for-predicting-stock-prices-2974c0c8c4ef
## 위의 링크 한글화 https://skyeong.net/309

import yfinance
import numpy as np
import os
from keras import callbacks
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc

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

class Predictor():
    def __init__(self):
        super().__init__()
        self.model = None
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None

        path = "c:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc('font', family=font_name)


        self.loadData()


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
        x, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence) - 1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x), np.array(y)

    def data_setup(self, n_steps, n_seq, sequence):
        x, y = self.split_sequences(sequence, n_steps)
        n_features = x.shape[2]
        x = x.reshape((len(x), n_steps, n_features))
        new_y = []
        for term in y:
            new_term = term[-1]
            new_y.append(new_term)
        return x, np.array(new_y), n_features

    def loadData(self):
        """
        야후 파이낸스에서 데이타를 읽어 온다.
        :return:
        """
        df = yfinance.download('AAPL', '2000-1-1', '2020-1-1')
        df = df.drop(['Volume'], 1).drop(['Adj Close'], 1)

        dataset, min_max = self.normalize_data(df)
        # print(df.values)
        """
        [[ 0.90122801  0.91852701  0.88839298  0.91796899]
         [ 0.93638402  1.00446403  0.907924    0.99944198]
         [ 0.96651798  0.98772299  0.90346003  0.91517901]
         ...
         [72.77999878 73.49250031 72.02999878 72.44999695]
         [72.36499786 73.17250061 71.30500031 72.87999725]
         [72.48249817 73.41999817 72.37999725 73.41249847]]
        """
        values = dataset.values

        n_steps = 10
        n_seq = 10000
        rel_test_len = 0.1
        x, y, n_features = self.data_setup(n_steps, n_seq, values)
        x = x[:-1]
        y = y[1:]

        print(len(x), int(len(x) * rel_test_len))
        self.x_test, self.y_test = x[:int(len(x) * rel_test_len)], y[:int(len(x) * rel_test_len)]
        self.x_train, self.y_train = x[int(len(x) * rel_test_len):], y[int(len(x) * rel_test_len):]

        print(self.x_test, self.y_test)
        print(x.shape)
        """
        (5021, 10, 4)
        """
    def lstm(self):
        self.model = Sequential()
        self.model.add(LSTM(64, activation=None, input_shape=(10, 4), return_sequences=True))
        self.model.add(LSTM(32, activation=None, return_sequences=True))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation=None))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='mse', optimizer='adam')

    def cnn(self):
        self.model = Sequential()
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

        history = self.model.fit(self.x_train,
                                 self.y_train,
                            epochs=epochs,
                            batch_size=len(self.x_train) // 4,
                            validation_data = (self.x_test, self.y_test),
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

    def drawGraph(self, **args):
        """
        그래프 출력
        :return: 
        """
        title = None
        if args['title']:
            title = args['title']

        self.model.evaluate(self.x_test, self.y_test)

        pred_test = self.model.predict(self.x_test)
        plt.plot(pred_test, 'r')
        plt.plot(self.y_test, 'g')
        plt.title(title)
        plt.legend(['예측값', '실제값'])
        plt.show()

if __name__ == "__main__":
    pred = Predictor()
    pred.lstm()
    # pred.train_networks('lstm')
    # pred.model = pred.load_keras_model('adam', 'lstm')
    # pred.drawGraph(title='LSTM에 의한 예측결과 ')

    # pred.lstm()
    # # pred.train_networks('cnn')
    # pred.model = pred.load_keras_model('adam', 'cnn')
    # pred.drawGraph(title='CNN에 의한 예측결과 ')

    
    # pred.byProphet('005930')  # 삼성전자 (005930)
    # pred.byProphet('253450')
    # pred.byProphet('011210')
    # pred.byProphet('000660') # sk 하이닉스
    # pred.byProphet('272210')
    # pred.byProphet('278280') # 천보
