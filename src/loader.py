import kagglehub
import pandas as pd
from stock import Stock

class Loader:
    def __init__(self):
        # ====================================================
        # Download datasets from kagglehub (Updated daily)
        # ====================================================
        path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")

        self.companies = pd.read_csv(f'{path}/sp500_companies.csv', 
            index_col="Symbol")
        
        self.stocks = self.refine_stock_data(
            pd.read_csv(f'{path}/sp500_stocks.csv', index_col="Date", parse_dates=["Date"])
        )
        self.index = pd.read_csv(f'{path}/sp500_index.csv', 
            index_col="Date", parse_dates=["Date"])

    def get_company_info(self, symbol: str) -> pd.DataFrame:
        return self.companies[self.companies.index == symbol]

    def get_rsi(self, data: pd.DataFrame, window) -> pd.DataFrame: 
        delta = data.diff()        

        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss

        return 100 - (100 / (1 + rs))

    def refine_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
            lambda x: self.get_rsi(x, 14))

        for column in ['SMA_low', 'SMA_high', 'EMA_low', 'EMA_high', 'RSI_val']:
            # First rows will always be Nan and needs to be
            # backfilled otherwise training will generate Nan metrics
            data[column] = data.groupby('Symbol')[column].bfill()
        return data

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        # ===========================================
        # Make sure data is sorted by Date
        # =========================================== 
        return self.stocks[self.stocks['Symbol'] == symbol]
        
    def get_stock(self, symbol: str) -> Stock:
        return Stock(self.get_company_info(symbol), self.get_stock_data(symbol))

    def get_symbols(self) -> list:
        return list(map(lambda x: self.get_stock(x), self.stocks['Symbol'].unique()))