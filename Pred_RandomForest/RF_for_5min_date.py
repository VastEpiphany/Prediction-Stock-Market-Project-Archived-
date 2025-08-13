import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# 加载数据
X_train = np.load("X_train_5.npy", allow_pickle=True)
y_train = np.load("y_train_5.npy", allow_pickle=True)
X_test = np.load("X_test_5.npy", allow_pickle=True)
y_test = np.load("y_test_5.npy", allow_pickle=True)

# 确保y_train和y_test是一维数组
y_train = y_train.ravel()
y_test = y_test.ravel()

# X_train的形状是[samples, timesteps, features]
# 将其重塑为[samples, timesteps * features]
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# 定义基础模型
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('svr', SVR())
]

# 创建堆叠集成模型
stack_reg = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# 训练模型
stack_reg.fit(X_train_reshaped, y_train)

# 进行预测
predictions = stack_reg.predict(X_test_reshaped)

# 计算RMSE
rmse = sqrt(mean_squared_error(y_test, predictions))
print("Stacking Ensemble RMSE:", rmse)

# 可视化预测结果
plt.figure(figsize=(16, 8))
plt.plot(y_test, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.title('Stock Price Prediction using Stacking Ensemble')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

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
