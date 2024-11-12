
import pandas as pd

# Simple Moving Average
def calculate_SMA(data: pd.Series, window: int) -> pd.Float64Dtype:
    return data.rolling(window=window).mean()

# Exponential Moving Average
def calculate_EMA(data: pd.Series, window: int) -> pd.Float64Dtype:
    return data.ewm(span=window, adjust=False).mean()

# Double Exponential Moving Average
def calculate_DEMA(data: pd.Series, window: int) -> pd.Float64Dtype:
    ema_1 = calculate_EMA(data, window)
    ema_2 = calculate_EMA(ema_1, window)
    return 2 * ema_1 - ema_2

# Relative strength index
def calculate_RSI(data: pd.Series, window: int) -> pd.Float64Dtype: 
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Rate of Change Ratio
def calculate_ROCR(data: pd.Series, window: int) -> pd.Float64Dtype: 
    return data / data.shift(window)

