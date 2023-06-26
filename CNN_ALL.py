import math
import datetime
import _strptime
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import keras.backend as bk
import tensorflow as tf
import warnings
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.layers import *
from sklearn.metrics import accuracy_score


# Predict future pre_window days price
pre_window = 5
# Stock Code
sk_code = '600660.XSHG'
# Get data start date
start_date = '2013-1-1'
# Get data end date
end_date = '2022-12-31'
# Training set ratio
train_ratio = 0.5
# Window time
prepare_window = 30
cnn_batch_size = 67
cnn_epoch = 70
cnn_learning_rate = 0.006
cnn_act_fun = 'relu'
cnn_ken_size = [2, 2]
cnn_str = [2, 2]
filter_num = 64
dropout_prob = 0.5
mp_size = [2, 2]
mp_str = [1, 1]
padding_p = 'same'
lstm_learning_rate = 0.004
lstm_epoch = 70
lstm_batch_size = 32
lstm_unit = 256
lstm_dropout = 0.3

# Import Data
l = os.listdir('/home/me/Desktop/Stock/Project_SP/csv')
for p in l:
    file_path = '/home/me/Desktop/Stock/Project_SP/csv/' + p
    data = pd.read_csv(file_path)
    dates = data[['Date']]
    data = data[['Open', 'High', 'Low', 'Close']]
    df = np.array(data)
    dates = np.array(dates)
    print(dates)
    print(type(dates))
    print(df.shape)

    # Data normalization and construction
    ts = math.floor(len(df)*train_ratio)
    x_train = df[:ts]
    y_train = df[:ts, -1]
    x_test = df[ts:]
    y_test = df[ts:, -1]

    dates = dates[ts + prepare_window + pre_window - 1:]

    datest = []
    for i in range (len(dates)):
        mylist = dates[i]
        list = mylist[0].split('/')
        list1 = []
        list1.append(list[2])
        list1.append(list[0])
        list1.append(list[1])
        #list1.append('')
        list2 = '-'.join(list1)
        datest.append(list2)
    print(datest)

    date = [datetime.strptime(d, '%Y-%m-%d').date() for d in datest]

    sx = MinMaxScaler(feature_range=(0, 1))
    sy = MinMaxScaler(feature_range=(0, 1))

    x_train_sca = sx.fit_transform(x_train)
    x_test_sca = sx.fit_transform(x_test)
    y_train_sca = sy.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test_sca = sy.fit_transform(np.array(y_test).reshape(-1, 1))

    x = []
    y = []
    xt = []
    yt = []
    yt_a = []

    # Construct the training set according to the 30-day prediction method
    for i in range(len(x_train_sca) - prepare_window - pre_window + 1):
        x.append(x_train_sca[i:i + prepare_window])
        y.append(y_train_sca[i + prepare_window + pre_window - 1][-1])
    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))


    for i in range(len(x_test_sca) - prepare_window - pre_window + 1):
        xt.append(x_test_sca[i:i + prepare_window])
        yt.append(y_test_sca[i + prepare_window + pre_window - 1][-1])
    xt = np.array(xt)
    print(xt.shape)
    yt = np.array(yt)
    print(yt.shape)
    xt = np.reshape(xt, (xt.shape[0], xt.shape[1], xt.shape[2]))

    for i in range(len(yt) - pre_window):
        if (yt[i + pre_window] - yt[i]) >= 0:
            yta = 1
        else:
            yta = 0
        yt_a.append(yta)
    yt_a = np.array(yt_a)

    #Construct convolutional neural network model, train and validate
    model = Sequential()
    model.add(Reshape((x.shape[1], x.shape[2], 1)))
    model.add(Conv2D(kernel_size=cnn_ken_size, strides=cnn_str, filters=filter_num, activation=cnn_act_fun, padding='same'))
    model.add(Conv2D(kernel_size=cnn_ken_size, strides=cnn_str, filters=filter_num, activation=cnn_act_fun, padding='same'))
    model.add(MaxPooling2D(pool_size=mp_size, strides=mp_str, padding=padding_p))
    model.add(Conv2D(kernel_size=cnn_ken_size, strides=cnn_str, filters=filter_num, activation=cnn_act_fun, padding='same'))
    model.add(MaxPooling2D(pool_size=mp_size, strides=mp_str, padding=padding_p))
    model.add(Flatten())
    model.add(Reshape((-1, 1)))
    model.add(LSTM(128))
    model.add(Dropout(dropout_prob))
    model.add(Dense(1))

    opt = tf.keras.optimizers.Adam(learning_rate=cnn_learning_rate)
    def scheduler(epoch):
        if epoch != 0 and epoch % 10 == 0:
            learning_rate = bk.get_value(model.optimizer.learning_rate)
            bk.set_value(model.optimizer.learning_rate, learning_rate * 0.7)
        return bk.get_value(model.optimizer.learning_rate)


    reduce_lr = LearningRateScheduler(scheduler)

    model.compile(optimizer=opt, loss='mean_squared_error')
    history = model.fit(x, y, batch_size=cnn_batch_size, epochs=cnn_epoch, validation_data=(xt, yt), shuffle=True, callbacks=[reduce_lr])

    predictions_c = model.predict(xt)
    yt_c = []
    for i in range(len(yt) - pre_window):
        if (predictions_c[i + pre_window] - predictions_c[i]) >= 0:
            ytc = 1
        else:
            ytc = 0
        yt_c.append(ytc)
    yt_c = np.array(yt_c)

    mse = mean_squared_error(yt, predictions_c)
    mae = mean_absolute_error(yt, predictions_c)
    r2 = r2_score(yt, predictions_c)
    acc = accuracy_score(yt_a[:len(yt_a) - pre_window], yt_c[pre_window:])

    print('\nthe convolutional neural network model result is :')
    print(f'MSE is : {mse:}')
    print(f'MAE is : {mae:}')
    print(f'R-Squared is : {r2:}')
    print(f'ACC is : {acc:}')

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Stock 000300 CNN Train Log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    yt = sy.inverse_transform(yt.reshape(-1,1))
    predictions_c = sy.inverse_transform(predictions_c.reshape(-1, 1))

    plt.figure(figsize=(20, 10))
    plt.plot(date, predictions_c, label='Predicted Price')
    plt.plot(date, yt, label='Real Price')
    #plt.title('Stock 000300 Price CNN Prediction')
    plt.title('Stock' + ' ' + os.path.splitext(p)[0] + ' ' + 'Price CNN Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    ds = math.floor(len(date)/9)
    plt.xticks(date[::ds])
    plt.legend()
    #plt.savefig('./'+ str(600109) + '.png')
    plt.show()