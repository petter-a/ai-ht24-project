import kagglehub
import pandas as pd
from stock import Stock

class Loader:
    def __init__(self):
        '''
        Download datasets from kagglehub (Updated daily)
        '''
        path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")

        self.companies = pd.read_csv(f'{path}/sp500_companies.csv', index_col="Symbol")
        self.stocks = pd.read_csv(f'{path}/sp500_stocks.csv', index_col="Date", parse_dates=["Date"])
        self.index = pd.read_csv(f'{path}/sp500_index.csv', index_col="Date", parse_dates=["Date"])

    def get_company_info(self, symbol: str) -> pd.DataFrame:
        return self.companies[self.companies.index == symbol]
    
    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        # ===========================================
        # Make sure data is sorted by Date
        # =========================================== 
        data = self.stocks[self.stocks['Symbol'] == symbol].sort_index()
        
        # ===========================================
        # Create the technical indicators
        # =========================================== 
        data['SMA_low'] = data['Close'].transform(
            lambda x: x.rolling(window=5).mean())
        
        data['SMA_high'] = data['Close'].transform(
            lambda x: x.rolling(window=10).mean())
        
        data['EMA_low'] = data['Close'].transform(
            lambda x: x.ewm(span=12, adjust=False).mean())

        data['EMA_high'] = data['Close'].transform(
            lambda x: x.ewm(span=26, adjust=False).mean())
        
        for column in ['SMA_low', 'SMA_high', 'EMA_low', 'EMA_high']:
            # First rows will always be Nan and needs to be
            # backfilled otherwise training will generate Nan metrics
            data[column] = data[column].fillna(method='bfill')
        return data

    def get_stock(self, symbol: str) -> Stock:
        return Stock(self.get_company_info(symbol), self.get_stock_data(symbol))
