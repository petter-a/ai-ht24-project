# Stock analysis

The purpose of the project is to provide insights into the stock market. The long term vision is to develop a web-based dashboard that highlights various buy and sell signals, combining predictions with technical analysis
to build a virtual portfolio with risk and value weigthed stocks.

# Features

Not all features might be made available at project completion date. Much focus will be spent on
the prediction part.

    * Stock statistics and price development for all stocks in the US S&P500 index.
    * Time period filtering
    * Stock price prediction
    * Basic technical analysis (Stock/Sectors/Index)
        * SMA (Simple moving average)
        * EMA (Exponential moving average)
        * RSI (Relative strength index)
    * Trending based on prediction
    * Risk
    * Top 10 stocks/sectors based on value increase over time
    * Buy/Sell signals

# Datasets

The primary data source is the [S&P 500 Stocks dataset](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks) available on Kaggle, which is updated daily. The dataset contains 14 years of daily historical data.

| Date       | Symbol | Adj Close  | Close      | High       | Low        | Open       | Volume    |
| ---------- | ------ | ---------- | ---------- | ---------- | ---------- | ---------- | --------- |
| 2024-10-17 | ZTS    | 191.000000 | 191.000000 | 196.550003 | 190.889999 | 195.710007 | 1701200.0 |
| 2024-10-18 | ZTS    | 193.279999 | 193.279999 | 193.490005 | 190.500000 | 191.160004 | 1576400.0 |
| 2024-10-21 | ZTS    | 189.449997 | 189.449997 | 193.000000 | 189.179993 | 192.479996 | 959500.0  |
| 2024-10-22 | ZTS    | 189.509995 | 189.509995 | 189.820007 | 187.220001 | 188.410004 | 1441900.0 |
| 2024-10-23 | ZTS    | 188.990005 | 188.990005 | 189.979996 | 187.559998 | 189.399994 | 1339482.0 |

Two additional auxiliary datasets are used to provide detailed company information (not visualized here) and index pricing history.

| Date       | S&P500  |
| ---------- | ------- |
| 2024-10-16 | 5842.47 |
| 2024-10-17 | 5841.47 |
| 2024-10-18 | 5864.67 |
| 2024-10-21 | 5853.98 |
| 2024-10-22 | 5851.20 |

# Method

Prediction is carried out using the LSTM (Long Short-Term Memory) neural network model, which is a preferred method for time series forecasting. Due to the volatility of the stock market, the model needs to be retrained quite frequently.
Initially based on a single feature (Closing price) but eventually more features like volume and volatility or other external data
that I can find that might strengthen the prognosis. An example is the SEK/USD exchange rates (which is available as download through
[Riksbanken](https://www.riksbank.se/en-gb/statistics/interest-rates-and-exchange-rates/search-interest-rates-and-exchange-rates/)).

# Challenges

The main challenge is to interpret and define thresholds for all parameters involved in the process.
For example what constitutes a "buy signal"? How to combine technical analysis (which in nature is statistical)
with predictions based on machine learning and what conclusions could one draw from it?
It is also a challenge to collect datasets with aligning time series in a consisten manner.
In the long run, various API's and static downloads needs to be combined using adapters to produce intermediate
datasets that can be combined efficiently.

These challenges will be adressed during the implementation phase.

# Scope of project

The planned deliverable for the project scope is a program that allows the user to choose a particular stock
from the S&P 500 index and retreive an analysis of the historical data and future predictions together with recommendations
given predefined thresholds.
