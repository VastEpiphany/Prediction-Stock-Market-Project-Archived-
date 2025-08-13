import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler

def preprocess_daily_data(filepath):
    # 读取日线数据
    with open(filepath, 'r') as file:
        column_names = file.readline().strip().split(',')
    df_day = pd.read_csv(filepath, delim_whitespace=True, skiprows=1, names=column_names)

    # 转换数据类型为数值型，对于无法转换的将会变为NaN
    df_day = df_day.apply(pd.to_numeric, errors='coerce')

    # 处理缺失值
    df_day.interpolate(method='linear', inplace=True)

    # 数据标准化
    scaler = StandardScaler()
    features_to_scale = ['Opening_Price', 'Highest_Price', 'Lowest_Price', 'Closing_Price', 'Trading_Volume']
    df_day[features_to_scale] = scaler.fit_transform(df_day[features_to_scale])

    # 对数变换，确保所有值都是正的
    df_day['Log_Closing_Price'] = np.log(df_day['Closing_Price'] + 1 - df_day['Closing_Price'].min())

    # Box-Cox变换
    df_day['BoxCox_Trading_Volume'] = boxcox(df_day['Trading_Volume'] + 1 - df_day['Trading_Volume'].min())[0]

    return df_day

def preprocess_minute_data(filepath):
    # 读取5分钟级数据
    with open(filepath, 'r') as file:
        column_names = file.readline().strip().split(',')
    df_minute = pd.read_csv(filepath, delim_whitespace=True, skiprows=1, names=column_names)

    # 转换数据类型为数值型
    df_minute = df_minute.apply(pd.to_numeric, errors='coerce')

    # 处理缺失值
    df_minute.interpolate(method='linear', inplace=True)

    # 移动平均 - 以5分钟数据为例
    df_minute['MA_5'] = df_minute['Closing_Price'].rolling(window=5).mean()

    # 数据标准化
    scaler = StandardScaler()
    features_to_scale = ['Opening_Price', 'Highest_Price', 'Lowest_Price', 'Closing_Price', 'Trading_Volume']
    df_minute[features_to_scale] = scaler.fit_transform(df_minute[features_to_scale])

    # 对数变换，确保所有值都是正的
    df_minute['Log_Closing_Price'] = np.log(df_minute['Closing_Price'] + 1 - df_minute['Closing_Price'].min())

    # Box-Cox变换
    df_minute['BoxCox_Trading_Volume'] = boxcox(df_minute['Trading_Volume'] + 1 - df_minute['Trading_Volume'].min())[0]

    return df_minute



# 使用例子
daily_data = preprocess_daily_data('daily_data.txt')
minute_data = preprocess_minute_data('5min_data.txt')
