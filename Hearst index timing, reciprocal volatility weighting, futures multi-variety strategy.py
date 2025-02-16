import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import akshare as ak
from hurst import compute_Hc

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 获取期货数据
def get_futures_data(symbol, start_date, end_date):
    futures_data = ak.futures_zh_daily_sina(symbol=symbol)
    futures_data['date'] = pd.to_datetime(futures_data['date'])
    futures_data.set_index('date', inplace=True)
    selected_data = futures_data.loc[start_date:end_date]
    close_prices = selected_data['close'].replace(0, np.nan).fillna(method='ffill')
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    return close_prices, log_returns

# 函数：计算策略的各项指标
def calculate_metrics(daily_returns):
    # 年化收益
    annual_return = np.power((1 + daily_returns).prod(), 252 / len(daily_returns)) - 1
    # 年化波动
    annual_volatility = daily_returns.std() * np.sqrt(252)
    # 夏普比率（假设无风险收益率为0）
    sharpe_ratio = annual_return / annual_volatility
    # 最大回撤
    cumulative = (1 + daily_returns).cumprod()
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()
    # 卡玛比率
    calmar_ratio = -annual_return / max_drawdown
    return annual_return, annual_volatility, sharpe_ratio, max_drawdown, calmar_ratio

# 函数：回测策略
def backtest_strategy(close_prices, log_returns, short_ma_length, long_ma_length, hurst_window_size, hurst_deviation_window, bollinger_window):
    # 初始化信号
    signals = pd.DataFrame(index=close_prices.index)
    signals['signal'] = 0

    # 计算均线
    short_ma = close_prices.rolling(window=short_ma_length).mean()
    long_ma = close_prices.rolling(window=long_ma_length).mean()

    # 计算布林带
    rolling_mean = close_prices.rolling(window=bollinger_window).mean()
    rolling_std = close_prices.rolling(window=bollinger_window).std()
    upper_band = rolling_mean + (bollinger_std_dev * rolling_std)
    lower_band = rolling_mean - (bollinger_std_dev * rolling_std)

    # 计算Hurst指数和偏差
    hurst_values = []
    historical_mean_hurst = []

    for start in range(len(close_prices) - hurst_window_size):
        window_data = close_prices[start:start + hurst_window_size]
        try:
            H, _, _ = compute_Hc(window_data, kind='price', simplified=False)
            hurst_values.append(H)
            if len(hurst_values) >= hurst_deviation_window:
                historical_mean_hurst.append(np.mean(hurst_values[-hurst_deviation_window:]))
            else:
                historical_mean_hurst.append(np.nan)
        except FloatingPointError:
            hurst_values.append(np.nan)
            historical_mean_hurst.append(np.nan)

    hurst_series = pd.Series(hurst_values, index=close_prices.index[hurst_window_size:])
    hurst_mean_series = pd.Series(historical_mean_hurst, index=close_prices.index[hurst_window_size:])
    hurst_deviation = hurst_series - hurst_mean_series

    # 根据Hurst偏差生成信号
    for i in range(1, len(hurst_deviation.index)):
        date = hurst_deviation.index[i]
        prev_date = hurst_deviation.index[i-1]

        if hurst_deviation[date] > 0:  # 正偏差，使用布林带
            # 检查上穿和下穿布林带的情况
            if close_prices[prev_date] <= lower_band[prev_date] and close_prices[date] > lower_band[date]:
                signals.at[date, 'signal'] = -1  # 看空信号，穿过下轨
            elif close_prices[prev_date] >= upper_band[prev_date] and close_prices[date] < upper_band[date]:
                signals.at[date, 'signal'] = 1  # 看多信号，穿过上轨
        elif hurst_deviation[date] < 0:  # 负偏差，使用双均线
            # 检查短期均线上穿和下穿长期均线的情况
            if short_ma[prev_date] <= long_ma[prev_date] and short_ma[date] > long_ma[date]:
                signals.at[date, 'signal'] = 1  # 看多信号，短线上穿长线
            elif short_ma[prev_date] >= long_ma[prev_date] and short_ma[date] < long_ma[date]:
                signals.at[date, 'signal'] = -1  # 看空信号，短线下穿长线

    # 处理交易信号，添加平仓逻辑
    position = 0
    hold_days = 0
    daily_returns_with_cost = []

    daily_returns = log_returns.loc[hurst_series.index]

    for i in range(1, len(daily_returns)):
        # 计算持仓收益
        if position != 0:
            hold_days += 1
            price_change = close_prices.iloc[i] / close_prices.iloc[i-1] - 1
            if position > 0 and (price_change > take_profit_threshold or price_change < -stop_loss_threshold or hold_days >= max_hold_days):
                position = 0  # 平仓
                hold_days = 0
            elif position < 0 and (-price_change > take_profit_threshold or -price_change < -stop_loss_threshold or hold_days >= max_hold_days):
                position = 0  # 平仓
                hold_days = 0

        # 更新仓位
        if signals['signal'].iloc[i] == 1 and position <= 0:
            position = 1
            hold_days = 0
        elif signals['signal'].iloc[i] == -1 and position >= 0:
            position = -1
            hold_days = 0
        
        # 记录每日收益（包含交易成本）
        daily_return = daily_returns.iloc[i] * position
        daily_return -= transaction_cost_rate * abs(position - signals['signal'].iloc[i-1])
        daily_returns_with_cost.append(daily_return)

    # 计算累积净值
    net_value = (1 + pd.Series(daily_returns_with_cost, index=daily_returns.index[1:])).cumprod()

    # 计算绩效指标
    annual_return, annual_volatility, sharpe_ratio, max_drawdown, calmar_ratio = calculate_metrics(pd.Series(daily_returns_with_cost))
    return annual_return, annual_volatility, sharpe_ratio, max_drawdown, calmar_ratio, net_value

# 设置起始和结束日期
start_date = '2014-08-15'
end_date = '2024-10-30'

# 固定策略参数
bollinger_std_dev = 1  # 布林带上下轨标准差倍数
take_profit_threshold = 0.05  # 止盈阈值（5%）
stop_loss_threshold = 0.05  # 止损阈值（5%）
max_hold_days = 50  # 最大持有期（50天）
transaction_cost_rate = 0.0001  # 交易成本率

# 品种及其参数设置
metal_params = {
    "ZN0": (100, 140, 140, 80, 20),
    "AL0": (80, 140, 140, 60, 60),
    "NI0": (20, 140, 140, 80, 80),
    "SN0": (100, 160, 140, 60, 80),
    "PB0": (100, 120, 160, 40, 60),
    "CU0": (80, 140, 140, 60, 60),
}

# 分层回测结果存储
portfolio_net_values = []
volatility_dataframes = {}

# 对每个金属品种进行回测
for symbol, params in metal_params.items():
    close_prices, log_returns = get_futures_data(symbol, start_date, end_date)
    short_ma_length, long_ma_length, hurst_window_size, hurst_deviation_window, bollinger_window = params
    
    _, _, _, _, _, net_value = backtest_strategy(
        close_prices, log_returns, short_ma_length, long_ma_length, hurst_window_size, hurst_deviation_window, bollinger_window
    )
    
    portfolio_net_values.append(net_value)
    
    # 计算波动率
    volatility_rolling = log_returns.rolling(window=160).std()  # fixed-length windows rolling
    volatility_extension = log_returns.expanding().std()  # extension
    
    # 将两个波动率合并，以便选择其中之一
    volatilities = pd.DataFrame({
        'rolling': volatility_rolling,
        'extension': volatility_extension
    }, index=log_returns.index).dropna()
    
    # 存储波动率数据
    volatility_dataframes[symbol] = volatilities

# 选择波动率计算方式：这里选择固定长度滚动波动率（可修改为 'extension'）
selected_volatility_data = pd.DataFrame({symbol: vol_df['rolling'] for symbol, vol_df in volatility_dataframes.items()})

# 对每个截面进行归一化
normalized_volatility = selected_volatility_data.div(selected_volatility_data.sum(axis=1), axis=0)

# 计算逆波动率加权
inverse_volatility_weights = 1 / normalized_volatility
inverse_volatility_weights = inverse_volatility_weights.div(inverse_volatility_weights.sum(axis=1), axis=0)

# 计算加权后的组合净值
weighted_returns = pd.DataFrame()

for i, (symbol, net_value) in enumerate(zip(metal_params.keys(), portfolio_net_values)):
    daily_returns = net_value.pct_change().fillna(0)
    weighted_returns[symbol] = daily_returns * inverse_volatility_weights[symbol]

combined_weighted_returns = weighted_returns.sum(axis=1)
combined_weighted_net_value = (1 + combined_weighted_returns).cumprod()

# 计算组合的绩效指标
annual_return, annual_volatility, sharpe_ratio, max_drawdown, calmar_ratio = calculate_metrics(combined_weighted_returns)

# 输出合成后的绩效指标
print(f"组合策略绩效指标：")
print(f"年化收益率: {annual_return:.2%}")
print(f"年化波动率: {annual_volatility:.2%}")
print(f"最大回撤: {max_drawdown:.2%}")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"卡玛比率: {calmar_ratio:.2f}")

# 绘制合成后的净值曲线
plt.figure(figsize=(12, 6))
plt.plot(combined_weighted_net_value, label='波动率加权组合策略净值', color='red')
plt.title('波动率加权组合策略净值曲线')
plt.xlabel('日期')
plt.ylabel('净值')
plt.legend()
plt.grid(True)
plt.show()
