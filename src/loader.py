import kagglehub
import pandas as pd
import config
import os
from stock import Stock
from typing import Self
class Loader:
    def __init__(self):
        self.companies = None
        self.stocks = None
        self.index = None

    def create_preprocessed_data(self, data_path: str = config.data_path) -> Self:
        # ====================================================
        # Download datasets from kagglehub (Updated daily)
        # ====================================================
        path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
        self.index = pd.read_csv(f'{path}/sp500_index.csv', index_col="Date", parse_dates=["Date"])
        self.companies = pd.read_csv(f'{path}/sp500_companies.csv', index_col="Symbol")        
        self.stocks = self.preprocess_stock_data(
            pd.read_csv(f'{path}/sp500_stocks.csv', index_col="Date", parse_dates=["Date"])
        )
        
        # ====================================================
        # Create output directory
        # ====================================================
        if not (os.path.exists(data_path) and os.path.isdir(data_path)):
            os.mkdir(data_path)
        
        # ====================================================
        # Cache preprocessed data
        # ====================================================
        # Preprocessing is a quite intence operation, flushing
        # the result to disk improves operations on subsequent
        # calls.

        self.index.to_csv(f'{data_path}/sp500_index.csv')
        self.companies.to_csv(f'{data_path}/sp500_companies.csv')
        self.stocks.to_csv(f'{data_path}/sp500_stocks.csv')
        return self

    def load_preprocessed_data(self, data_path: str = config.data_path) -> Self:
        # ====================================================
        # Read preprocessed data from file
        # ====================================================
        self.index = pd.read_csv(f'{data_path}/sp500_index.csv', index_col="Date", parse_dates=["Date"])
        self.companies = pd.read_csv(f'{data_path}/sp500_companies.csv', index_col="Symbol")
        self.stocks = pd.read_csv(f'{data_path}/sp500_stocks.csv', index_col="Date", parse_dates=["Date"])
        return self

    def get_company_info(self, symbol: str) -> pd.DataFrame:
        return self.companies[self.companies.index == symbol]

    def list_companies(self) -> list:
        return self.companies.sort_index().to_string(columns=['Shortname'])

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        # ===========================================
        # Make sure data is sorted by Date
        # =========================================== 
        return self.stocks[self.stocks['Symbol'] == symbol]
        
    def get_stock(self, symbol: str) -> Stock:
        return Stock(self.get_company_info(symbol), self.get_stock_data(symbol))

    def get_symbols(self) -> list:
        return list(map(lambda x: self.get_stock(x), self.stocks['Symbol'].unique()))
    
    def preprocess_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # ===========================================
        # Make sure data is sorted by Date
        # =========================================== 
        data = data.sort_index().dropna()        
        # ===========================================
        # Create the technical indicators
        # =========================================== 
        data['SMA_low'] = data.groupby('Symbol')['Adj Close'].transform(
            lambda x: x.rolling(window=50).mean())
        
        data['SMA_high'] = data.groupby('Symbol')['Adj Close'].transform(
            lambda x: x.rolling(window=200).mean())
        
        data['EMA_low'] = data.groupby('Symbol')['Adj Close'].transform(
            lambda x: x.ewm(span=12, adjust=False).mean())

        data['EMA_high'] = data.groupby('Symbol')['Adj Close'].transform(
            lambda x: x.ewm(span=36, adjust=False).mean())

        data['RSI_val'] = data.groupby('Symbol')['Adj Close'].transform(
            lambda x: self.calculate_rsi(x, 14))

        for column in ['SMA_low', 'SMA_high', 'EMA_low', 'EMA_high', 'RSI_val']:
            # First rows will always be Nan and needs to be
            # backfilled otherwise training will generate Nan metrics
            # In cases where backfill cannot take place (A coluymn with only NaN)
            # Fallback to '0'
            data[column] = data.groupby('Symbol')[column].bfill().fillna(0)

        return data

    def calculate_rsi(self, data: pd.DataFrame, window: int) -> pd.DataFrame: 
        delta = data.diff()        

        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
