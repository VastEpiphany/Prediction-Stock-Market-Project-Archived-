import os
import csv
import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
import statsmodels.api as sm
from numpy import *
from math import sqrt
from pandas import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from matplotlib.dates import DateFormatter

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
# 指定您的TXT和CSV文件路径
input_file = '5min_data.txt'
output_file = '5min_data.csv'

# 打开TXT文件和目标CSV文件
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    # 创建CSV写入器
    writer = csv.writer(outfile)

    for line in infile:
        # 将每行分割为列表，这里假设分隔符是制表符'\t'
        row = line.strip().split(',')
        writer.writerow(row)
        break

    # 逐行读取TXT文件，并写入CSV文件
    for line in infile:
        # 将每行分割为列表，这里假设分隔符是制表符'\t'
        row = line.strip().split('\t')
        writer.writerow(row)

print("数据已成功写入", output_file)

input_file = 'daily_data.txt'
output_file = 'daily_data.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    # 创建CSV写入器
    writer = csv.writer(outfile)

    for line in infile:
        # 将每行分割为列表，这里假设分隔符是制表符'\t'
        row = line.strip().split(',')
        writer.writerow(row)
        break

    # 逐行读取TXT文件，并写入CSV文件
    for line in infile:
        # 将每行分割为列表，这里假设分隔符是制表符'\t'
        row = line.strip().split('\t')
        writer.writerow(row)

print("数据已成功写入", output_file)


## import data
df = pd.read_csv('daily_data.csv', parse_dates=['Date'])
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.columns)  # 打印列名
print(df.index)    # 打印索引

df_ = pd.read_csv('5min_data.csv', parse_dates=['Date', 'Time'])
print(df_.head())
print(df_.tail())
print(df_.shape)
print(df_.columns)  # 打印列名
print(df_.index)    # 打印索引

# Create Apple stock price plot
## https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series-data-in-python/date-time-types-in-pandas-python/customize-dates-matplotlib-plots-python/
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(df['Date'], df['Closing_Price'] ,label='Stock Price')
ax.set(xlabel="Date",
       ylabel="CNY",
       title="Stock Price")
date_form = DateFormatter("%Y")
ax.xaxis.set_major_formatter(date_form)
plt.show()

fig, ax = plt.subplots(figsize=(10,3))
df_['Date'] = df_['Date'].astype(str)
df_['Time'] = df_['Time'].astype(str)
df_['DateTime'] = pd.to_datetime(df_['Date'] +' ' + df_['Time'])
ax.plot(df_['DateTime'], df_['Closing_Price'] ,label='Stock Price')
ax.set(xlabel="DateTime",
       ylabel="CNY",
       title="Stock Price")
date_form = DateFormatter("%Y")
ax.xaxis.set_major_formatter(date_form)
plt.show()

dataset = pd.read_csv('daily_data.csv', parse_dates=['Date'])
# Replace 0 by NA
dataset.replace(0, np.nan, inplace=True)
dataset.to_csv("dataset.csv", index=False)

dataset_5 = pd.read_csv('5min_data.csv', parse_dates=['Date', 'Time'])
# Replace 0 by NA
dataset_5.replace(0, np.nan, inplace=True)
dataset_5.to_csv("dataset_5.csv", index=False)

# Set the date to datetime data
datetime_series = pd.to_datetime(dataset['Date'])
datetime_index = pd.DatetimeIndex(datetime_series.values)
dataset = dataset.set_index(datetime_index)
dataset = dataset.sort_values(by='Date')
dataset = dataset.drop(columns='Date')

# Set the date to datetime data
dataset_5['Date'] = df_['Date'].astype(str)
dataset_5['Time'] = df_['Time'].astype(str)
dataset_5['DateTime'] = pd.to_datetime(df_['Date'] +' ' + df_['Time'])
datetime_series_5 = pd.to_datetime(dataset_5['DateTime'])
datetime_index_5 = pd.DatetimeIndex(datetime_series_5.values)
dataset_5 = dataset_5.set_index(datetime_index_5)
dataset_5 = dataset_5.sort_values(by='DateTime')
#dataset_5['Timestamp'] = dataset_5['DateTime'].astype('int64')
dataset_5 = dataset_5.drop(columns='DateTime')
dataset_5 = dataset_5.drop(columns='Date')
dataset_5 = dataset_5.drop(columns='Time')

# Get features and target
X_value = pd.DataFrame(dataset.iloc[:, :4])
y_value = pd.DataFrame(dataset.iloc[:, 3])

# Get features and target
X_value_5 = pd.DataFrame(dataset_5.iloc[:, :4])
y_value_5 = pd.DataFrame(dataset_5.iloc[:, 3])

print(X_value)
print(y_value)
print(X_value_5)
print(y_value_5)
# Normalized the data
X_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

X_scaler.fit(X_value)
y_scaler.fit(y_value)

X_scale_dataset = X_scaler.fit_transform(X_value)
y_scale_dataset = y_scaler.fit_transform(y_value)

dump(X_scaler, open('X_scaler.pkl', 'wb'))
dump(y_scaler, open('y_scaler.pkl', 'wb'))
# -------------------------------------------------------------------------------------
X_scaler_5 = MinMaxScaler(feature_range=(-1, 1))
y_scaler_5 = MinMaxScaler(feature_range=(-1, 1))

X_scaler_5.fit(X_value_5)
y_scaler_5.fit(y_value_5)

X_scale_dataset_5 = X_scaler_5.fit_transform(X_value_5)
y_scale_dataset_5 = y_scaler_5.fit_transform(y_value_5)

dump(X_scaler_5, open('X_scaler_5.pkl', 'wb'))
dump(y_scaler_5, open('y_scaler_5.pkl', 'wb'))

n_steps_in = 3
n_features = X_value.shape[1]
n_features_5 = X_value_5.shape[1]
n_steps_out = 1

def get_X_y(X_data, y_data):
    X = list()
    y = list()
    yc = list()

    length = len(X_data)
    for i in range(0, length, 1):
        X_value = X_data[i: i + n_steps_in][:, :]
        y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
        yc_value = y_data[i: i + n_steps_in][:, :]
        if len(X_value) == 3 and len(y_value) == 1:
            X.append(X_value)
            y.append(y_value)
            yc.append(yc_value)

    return np.array(X), np.array(y), np.array(yc)


# get the train test predict index
def predict_index(dataset, X_train, n_steps_in, n_steps_out):

    # get the predict data (remove the in_steps days)
    train_predict_index = dataset.iloc[n_steps_in : X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index
    test_predict_index = dataset.iloc[X_train.shape[0] + n_steps_in:, :].index

    return train_predict_index, test_predict_index

# Split train/test dataset
def split_train_test(data):
    train_size = round(len(data) * 0.7)
    data_train = data[0:train_size]
    data_test = data[train_size:]
    return data_train, data_test

# Get data and check shape
X, y, yc = get_X_y(X_scale_dataset, y_scale_dataset)
X_train, X_test, = split_train_test(X)
y_train, y_test, = split_train_test(y)
yc_train, yc_test, = split_train_test(yc)
index_train, index_test, = predict_index(dataset, X_train, n_steps_in, n_steps_out)

X_5, y_5, yc_5 = get_X_y(X_scale_dataset_5, y_scale_dataset_5)
X_train_5, X_test_5, = split_train_test(X_5)
y_train_5, y_test_5, = split_train_test(y_5)
yc_train_5, yc_test_5, = split_train_test(yc_5)
index_train_5, index_test_5, = predict_index(dataset_5, X_train_5, n_steps_in, n_steps_out)
# %% --------------------------------------- Save dataset -----------------------------------------------------------------
print('X shape: ', X.shape)
print('y shape: ', y.shape)
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('y_c_train shape: ', yc_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)
print('y_c_test shape: ', yc_test.shape)
print('index_train shape:', index_train.shape)
print('index_test shape:', index_test.shape)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
np.save("yc_train.npy", yc_train)
np.save("yc_test.npy", yc_test)
np.save('train_predict_index.npy', index_train)
np.save('test_predict_index.npy', index_test)

print('X_5 shape: ', X_5.shape)
print('y_5 shape: ', y_5.shape)
print('X_train_5 shape: ', X_train_5.shape)
print('y_train_5 shape: ', y_train_5.shape)
print('y_c_train_5 shape: ', yc_train_5.shape)
print('X_test_5 shape: ', X_test_5.shape)
print('y_test_5 shape: ', y_test_5.shape)
print('y_c_test_5 shape: ', yc_test_5.shape)
print('index_train_5 shape:', index_train_5.shape)
print('index_test_5 shape:', index_test_5.shape)

np.save("X_train_5.npy", X_train_5)
np.save("y_train_5.npy", y_train_5)
np.save("X_test_5.npy", X_test_5)
np.save("y_test_5.npy", y_test_5)
np.save("yc_train_5.npy", yc_train_5)
np.save("yc_test_5.npy", yc_test_5)
np.save('train_predict_index_5.npy', index_train_5)
np.save('test_predict_index_5.npy', index_test_5)


