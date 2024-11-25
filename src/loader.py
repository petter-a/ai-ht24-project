import kagglehub
import pandas as pd
import config
import os
import indicators
from stock import Stock
from typing import Self
class Loader:
    def __init__(self):
        self.companies = None
        self.stocks = None
        self.index = None

    def create_preprocessed_data(self, cache_path: str = config.cache_path) -> Self:
        print('Downloading datasets ....')
        # ====================================================
        # Download datasets from kagglehub (Updated daily)
        # ====================================================
        path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")

        # ====================================================
        # Read and pre-process data
        # ====================================================
        self.index = pd.read_csv(
            f'{path}/sp500_index.csv', 
            index_col="Date", 
            parse_dates=["Date"]
        )
        self.companies = pd.read_csv(
            f'{path}/sp500_companies.csv', 
            index_col="Symbol"
        )        
        stock_data = pd.read_csv(
            f'{path}/sp500_stocks.csv', 
            index_col="Date", 
            parse_dates=["Date"]
        )
        self.stocks = self.preprocess_stock_data(stock_data)
        
        # ====================================================
        # Create output directory
        # ====================================================
        if not (os.path.exists(cache_path) and os.path.isdir(cache_path)):
            os.mkdir(cache_path)
        
        # ====================================================
        # Cache preprocessed data
        # ====================================================
        # Preprocessing is a quite intence operation, flushing
        # the result to disk improves operations on subsequent
        # calls.

        self.index.to_csv(f'{cache_path}/sp500_index.csv')
        self.companies.to_csv(f'{cache_path}/sp500_companies.csv')
        self.stocks.to_csv(f'{cache_path}/sp500_stocks.csv')
        return self

    def load_preprocessed_data(self, cache_path: str = config.cache_path) -> Self:
        print('Loading data from cache ....')
        # ====================================================
        # Read preprocessed data from file
        # ====================================================
        self.index = pd.read_csv(f'{cache_path}/sp500_index.csv', index_col="Date", parse_dates=["Date"])
        self.companies = pd.read_csv(f'{cache_path}/sp500_companies.csv', index_col="Symbol")
        self.stocks = pd.read_csv(f'{cache_path}/sp500_stocks.csv', index_col="Date", parse_dates=["Date"])
        return self

    def get_company_info(self, symbol: str) -> pd.DataFrame:
        return self.companies[self.companies.index == symbol]

    def list_companies(self) -> list:
        return self.companies.sort_index().to_string(columns=['Shortname'])

    def get_stock_data(self, symbol: str = None) -> pd.DataFrame:
        if None != symbol:
            return self.stocks[self.stocks['Symbol'] == symbol]
        return self.stocks
        
    def get_stock(self, symbol: str) -> Stock:
        return Stock(self.get_company_info(symbol), self.get_stock_data(symbol))
    
    def preprocess_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        print('Pre-processing data ....')
        # ===========================================
        # Make sure data is sorted by Date
        # =========================================== 
        data = data.sort_values(by=['Symbol', 'Date']).dropna()
        # ===========================================
        # Create the technical indicators
        # =========================================== 
        close = data.groupby('Symbol')['Close']

        data['SMA_low'] = close.transform(
            lambda x: indicators.calculate_SMA(x, 50))
        
        data['SMA_high'] = close.transform(
            lambda x: indicators.calculate_SMA(x, 200))
        
        data['EMA_low'] = close.transform(
            lambda x: indicators.calculate_EMA(x, 12))

        data['EMA_high'] = close.transform(
            lambda x: indicators.calculate_EMA(x, 36))

        data['RSI_val'] = close.transform(
            lambda x: indicators.calculate_RSI(x, 14))

        data['DEMA_val'] = close.transform(
            lambda x: indicators.calculate_DEMA(x, 14))

        data['ROCR_val'] = close.transform(
            lambda x: indicators.calculate_ROCR(x, 5))

        data['HURST_val'] = close.transform(
            lambda x: indicators.calculate_HURST(x.values, 50))

        for column in ['SMA_low', 'SMA_high', 'EMA_low', 'EMA_high', 'RSI_val', 'DEMA_val', 'ROCR_val', 'HURST_val']:
            # First rows will always be Nan and needs to be
            # backfilled otherwise training will generate Nan metrics
            # In cases where backfill cannot take place (A column with only NaN)
            # Fallback to '0'
            data[column] = data.groupby('Symbol')[column].bfill().fillna(0)

        return data