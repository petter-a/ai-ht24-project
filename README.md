# Stock analysis

The purpose of the project is to provide technical analysis and stock price predictions.

# Features

    * Stock statistics and price development for all stocks in the US S&P500 index.
    * Time period filtering
    * Stock price prediction
    * Basic technical analysis (Stock/Sectors/Index)
        * SMA
        * EMA
    * Top 10 stocks/sectors based on value increase over time
    * Buy/Sell signals

# Datasets

The main data source is the kagglehub "sp-500-stocks" dataset,
a dataset which figures are updated daily. The data extends 24 years of daily history:

| Date       | Symbol | Adj Close  | Close      | High       | Low Open   | Volume     |
| ---------- | ------ | ---------- | ---------- | ---------- | ---------- | ---------- | --------- |
| 2024-10-17 | ZTS    | 191.000000 | 191.000000 | 196.550003 | 190.889999 | 195.710007 | 1701200.0 |
| 2024-10-18 | ZTS    | 193.279999 | 193.279999 | 193.490005 | 190.500000 | 191.160004 | 1576400.0 |
| 2024-10-21 | ZTS    | 189.449997 | 189.449997 | 193.000000 | 189.179993 | 192.479996 | 959500.0  |
| 2024-10-22 | ZTS    | 189.509995 | 189.509995 | 189.820007 | 187.220001 | 188.410004 | 1441900.0 |
| 2024-10-23 | ZTS    | 188.990005 | 188.990005 | 189.979996 | 187.559998 | 189.399994 | 1339482.0 |

To additional helper datasets are used to supply detailed companyinformation (not visualized here) and index pricing history.

| Date       | S&P500  |
| ---------- | ------- |
| 2024-10-16 | 5842.47 |
| 2024-10-17 | 5841.47 |
| 2024-10-18 | 5864.67 |
| 2024-10-21 | 5853.98 |
| 2024-10-22 | 5851.20 |

# Method

Prediction is performed using the LSTM (long short-term memory) neural network model.
The dataset is updated every 24h with new predictions.
