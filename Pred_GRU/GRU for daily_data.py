
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import *
from math import sqrt
from pandas import *
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Embedding, TimeDistributed, LeakyReLU
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot
from pickle import load

# Load data
X_train = np.load("X_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
X_test = np.load("X_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)
yc_train = np.load("yc_train.npy", allow_pickle=True)
yc_test = np.load("yc_test.npy", allow_pickle=True)

# Parameters
LR = 0.0001
BATCH_SIZE = 128
N_EPOCH = 50

input_dim = X_train.shape[1]
feature_size = X_train.shape[2]
output_dim = y_train.shape[1]
print(size(X_train))
print(size(y_train))


def basic_GRU(input_dim, output_dim, feature_size) -> tf.keras.models.Model:
    model = Sequential()
    model.add(GRU(units=128, return_sequences = True, input_shape=(input_dim, feature_size)))  # 256, return_sequences = True
    # model.add(GRU(units=256, recurrent_dropout = 0.2)) #, return_sequences = True
    model.add(GRU(units=64, input_shape=(input_dim, feature_size)))
    #model.add(Dense(128))
    model.add(Dense(32))
    # model.add(Dense(32))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(lr=LR), loss='mse')
    history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                        verbose=2, shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()

    return model


model = basic_GRU(input_dim, output_dim, feature_size)
print(model.summary())
model.save('GRU_30to3.h5')

yhat = model.predict(X_test, verbose=0)
# print(yhat)

rmse = sqrt(mean_squared_error(y_test, yhat))
print("RMSE", rmse)
mse = mean_squared_error(y_test, yhat)
print("MSE", mse)


# %% --------------------------------------- Plot the TEST result  -----------------------------------------------------------------
def calculate_drawdowns(prices):
    """
    计算回撤和最大回撤。

    :param prices: 包含价格数据的 Pandas Series。
    :return: 包含每个时间点的回撤的 Pandas Series，以及一个最大回撤的值。
    """
    # 计算累积最大值
    cumulative_max = prices.cummax()
    # 计算回撤
    drawdowns = (prices - cumulative_max) / cumulative_max
    # 计算最大回撤
    max_drawdown = drawdowns.min()

    return drawdowns, max_drawdown


def plot_testdataset_result(X_test, y_test):

    test_yhat = model.predict(X_test, verbose=0)
    y_scaler = load(open('y_scaler.pkl', 'rb'))
    test_predict_index = np.load("test_predict_index.npy", allow_pickle=True)

    rescaled_real_y = y_scaler.inverse_transform(y_test)
    rescaled_predicted_y = y_scaler.inverse_transform(test_yhat)

    predict_result = pd.DataFrame()
    for i in range(rescaled_predicted_y.shape[0]):
        print("Plot test dataset result...")
        y_predict = pd.DataFrame(rescaled_predicted_y[i], columns=["predicted_price"],
                                 index=test_predict_index[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_price = pd.DataFrame()
    for i in range(rescaled_real_y.shape[0]):
        print("Plot test dataset result...")
        y_train = pd.DataFrame(rescaled_real_y[i], columns=["real_price"],
                               index=test_predict_index[i:i + output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)


    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)
    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_price["real_mean"])
    plt.plot(predict_result["predicted_mean"], color='r')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title("The result of Testing", fontsize=20)
    plt.show()

    real_series = pd.Series( real_price['real_mean'])
    predict_series = pd.Series(predict_result['predicted_mean'])
    real_drawdowns, real_max_drawdown = calculate_drawdowns(real_series)
    print("Real Max Drawdown:", real_max_drawdown)

    predicted_drawdowns, predicted_max_drawdown = calculate_drawdowns(predict_series)
    print("Predicted Max Drawdown:", predicted_max_drawdown)

    # 可视化回撤
    plt.figure(figsize=(16, 8))
    plt.plot(real_drawdowns, label='Real Price Drawdowns')
    plt.title("Drawdowns over Time")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.show()

    # 可视化回撤
    plt.figure(figsize=(16, 8))
    plt.plot(real_drawdowns, label='Prediction Price Drawdowns')
    plt.title("Drawdowns over Time")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.show()
    # Calculate RMSE
    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    RMSE = np.sqrt(mean_squared_error(predicted, real))

    return RMSE

test_RMSE = plot_testdataset_result(X_test, y_test)
print("----- Test_RMSE -----", test_RMSE)