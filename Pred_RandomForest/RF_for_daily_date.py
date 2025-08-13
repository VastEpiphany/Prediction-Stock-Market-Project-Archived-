import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from pickle import load

# Load data
X_train = np.load("X_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
X_test = np.load("X_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)

# 确保y_train和y_test是一维数组
y_train = y_train.ravel()
y_test = y_test.ravel()

X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# 使用重塑后的数据训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_reshaped, y_train)

# 使用重塑后的测试数据进行预测
predictions = rf_model.predict(X_test_reshaped)

# 计算RMSE
rmse = sqrt(mean_squared_error(y_test, predictions))
print("Random Forest RMSE:", rmse)


# 可视化预测结果
def plot_predictions(y_test, predictions):
    plt.figure(figsize=(16, 8))
    plt.plot(y_test, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title('Stock Price Prediction using Random Forest')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


plot_predictions(y_test, predictions)


# 计算和绘制回撤
def calculate_drawdowns(prices):
    cumulative_max = prices.cummax()
    drawdowns = (prices - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()
    return drawdowns, max_drawdown


def plot_drawdowns(y_test, predictions):
    real_price_series = pd.Series(y_test)
    predicted_price_series = pd.Series(predictions)

    real_drawdowns, real_max_drawdown = calculate_drawdowns(real_price_series)
    predicted_drawdowns, predicted_max_drawdown = calculate_drawdowns(predicted_price_series)

    plt.figure(figsize=(16, 8))
    plt.plot(real_drawdowns, label='Real Price Drawdowns')
    plt.plot(predicted_drawdowns, label='Predicted Price Drawdowns')
    plt.title("Drawdowns over Time")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.show()

    print("Real Max Drawdown:", real_max_drawdown)
    print("Predicted Max Drawdown:", predicted_max_drawdown)


plot_drawdowns(y_test, predictions)
