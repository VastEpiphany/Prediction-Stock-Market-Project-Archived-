import pandas as pd
import numpy as np
from daily_data_model import daily_data_list 
from min_data_model import fivemin_data_list


# 日线数据：日期和预测的收盘值
daily_data = daily_data_list  

# 5分钟级数据：日期和每5分钟的预测收盘值
five_minute_data = fivemin_data_list

# 将数据转换为Pandas DataFrame
df_daily = pd.DataFrame(daily_data, columns=['Date', 'Predicted_Close'])
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_5min = pd.DataFrame(five_minute_data, columns=['Datetime', 'Predicted_Close'])
df_5min['Datetime'] = pd.to_datetime(df_5min['Datetime'])

# 收益率: 0.0008260451497673057
# 最大回撤: 0.020161040464376345
# # 参数设置
# buy_threshold = 0.05  # 买入阈值
# sell_threshold = 0.005  # 卖出阈值
# fixed_trade_amount = 10000  # 固定交易规模
# stop_loss_pct = 0.0075  # 止损点
# stop_gain_pct = 0.075  # 止盈点

# 参数设置
buy_threshold = 0.05  # 买入阈值
sell_threshold = 0.005  # 卖出阈值
fixed_trade_amount = 10000  # 固定交易规模
stop_loss_pct = 0.0075  # 止损点
stop_gain_pct = 0.075  # 止盈点


# 初始化账户信息
initial_capital = 1000000  # 初始资金：1000000元
capital = initial_capital  # 当前资金
position = 0  # 持仓量

# 模拟交易函数
def simulate_trading(df_daily, df_5min):
    global capital, position
    # 交易记录
    trade_records = []

    # 遍历每一天的5分钟级数据
    for date in df_daily['Date']:
        daily_close = df_daily[df_daily['Date'] == date]['Predicted_Close'].iloc[0]
        df_today = df_5min[df_5min['Datetime'].dt.date == date.date()]

        for _, row in df_today.iterrows():
            current_price = row['Predicted_Close']
            current_time = row['Datetime']

            # 买入逻辑
            if current_price > daily_close * (1 + buy_threshold) and capital >= fixed_trade_amount:
                # 执行买入
                buy_price = current_price
                capital -= fixed_trade_amount
                position += fixed_trade_amount / buy_price
                trade_records.append((current_time, 'BUY', buy_price, fixed_trade_amount))

            # 卖出逻辑
            elif current_price < daily_close * (1 - sell_threshold) and position > 0:
                # 执行卖出
                sell_price = current_price
                sell_amount = position * sell_price
                capital += sell_amount
                position = 0
                trade_records.append((current_time, 'SELL', sell_price, sell_amount))

            # 止损和止盈检查
            if position > 0:
                holding_value = position * current_price
                if (holding_value - fixed_trade_amount) / fixed_trade_amount <= -stop_loss_pct:
                    # 止损卖出
                    sell_price = current_price
                    sell_amount = position * sell_price
                    capital += sell_amount
                    position = 0
                    trade_records.append((current_time, 'STOP LOSS', sell_price, sell_amount))
                elif (holding_value - fixed_trade_amount) / fixed_trade_amount >= stop_gain_pct:
                    # 止盈卖出
                    sell_price = current_price
                    sell_amount = position * sell_price
                    capital += sell_amount
                    position = 0
                    trade_records.append((current_time, 'TAKE PROFIT', sell_price, sell_amount))

    return trade_records

# 运行模拟交易
trade_records = simulate_trading(df_daily, df_5min)

# # 显示部分交易记录
# print (trade_records[:10]) # 显示前10条记录

# 将交易记录转换为DataFrame
df_trades = pd.DataFrame(trade_records, columns=['Datetime', 'Type', 'Price', 'Amount'])
df_trades['Datetime'] = pd.to_datetime(df_trades['Datetime'])

# 计算资产净值（Net Asset Value, NAV）
capital = initial_capital
nav = [capital]
for _, trade in df_trades.iterrows():
    if trade['Type'] == 'BUY':
        capital -= trade['Amount']
    else:
        capital += trade['Amount']
    nav.append(capital)

df_trades['NAV'] = nav[1:]  # 将计算得到的NAV值添加到交易记录DataFrame中

# 计算收益率
final_nav = nav[-1]
return_rate = (final_nav - initial_capital) / initial_capital

# 计算最大回撤
max_drawdown = 0
peak = initial_capital
for val in nav:
    if val > peak:
        peak = val
    drawdown = (peak - val) / peak
    if drawdown > max_drawdown:
        max_drawdown = drawdown

# 输出结果
print("收益率:", return_rate)
print("最大回撤:", max_drawdown)

