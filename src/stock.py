import numpy as np
import pandas as pd
from config import DateRange
from model import StockModel

def to_str(frame: pd.DataFrame, field: str) -> str: return frame[field].values[0]

class Stock:
    def __init__(self, company: pd.DataFrame, data: pd.DataFrame):
        self.company = company
        self.data = data

    def get_company_name(self) -> str:
        return to_str(self.company, 'Shortname')
    
    def get_industry_name(self) -> str:
        return to_str(self.company, 'Industry')

    def get_sector_name(self) -> str:
        return to_str(self.company, 'Sector')

    def get_symbol_name(self) -> str:
        return self.company.index[0]

    def get_data_range(self, data: pd.DataFrame, range: DateRange = None) -> pd.DataFrame:
        result = data
        if range != None:
            if range[0] != None:
                result = result[result.index >= range[0] ]
            if range[1] != None:
                result = result[result.index <= range[1] ]
        return result

    def get_average_price(self, range: DateRange = None) -> pd.Float64Dtype:
        return np.average(self.get_data_range(self.data, range)['Adj Close'])
    
    def get_price(self, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data, range)['Adj Close']

    def get_volume(self, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data, range)["Volume"]

    def get_predicted_price(self) -> pd.DataFrame:
        p = StockModel(self.data, self.get_symbol_name())
        p.load_model()  
        return p.predict()
    
    def get_sma_high(self, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data['SMA_high'], range)

    def get_sma_low(self, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data['SMA_low'], range)

    def get_ema_high(self, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data['EMA_high'], range)

    def get_ema_low(self, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data['EMA_low'], range)

    def get_rsi_val(self, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data['RSI_val'], range)

    def get_closing_date(self) -> pd.Timestamp:
        return max(self.data.index)
    
    def get_closing_price(self) -> str:
        return self.data.iloc[-1]['Adj Close']
    
    def get_data(self) -> pd.DataFrame: return self.data