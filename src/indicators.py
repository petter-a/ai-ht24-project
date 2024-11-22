
import pandas as pd
import numpy as np

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

# Hurst exponent and fractal dimension
def calculate_HURST(data: pd.array, window:int, max_lag:int = 20):
    hurst_vals = []

    def hurst_exponent(time_series, max_lag):
        lags = range(2, max_lag + 1)
        tau = []
        for lag in lags:
            diff = time_series[lag:] - time_series[:-lag]
            tau.append(np.std(diff))
        tau = np.array(tau) + 1e-10
        hurst, _ = np.polyfit(np.log(lags), np.log(tau), 1)
        return hurst * 2.0

    for i in range(len(data)):
        if i < window:
            hurst_vals.append(np.nan)
        else:
            window_data = data[i-window:i]
            hurst = hurst_exponent(window_data, max_lag=max_lag)
            hurst_vals.append(hurst)
    return np.array(hurst_vals)