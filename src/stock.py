import numpy as np
import pandas as pd
from lib_types import DateRange
from model import StockModel

def to_str(frame: pd.DataFrame, field: str) -> str: return frame[field].values[0]

class Stock:
    def __init__(self, company: pd.DataFrame, data: pd.DataFrame):
        self.company = company
        self.data = data
        self.close = 'Close'

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
        return np.average(self.get_data_range(self.data, range)[self.close])
    
    def get_price(self, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data, range)[self.close]

    def get_volume(self, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data, range)["Volume"]

    def get_predicted_price(self) -> pd.DataFrame:
        p = StockModel(self.data)
        p.fromFile()  
        return p.predict()
    
    def get_sma(self, days: int, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data[self.close].rolling(days).mean(), range)

    def get_ema(self, days: int, range: DateRange = None) -> pd.Series:
        return self.get_data_range(self.data[self.close].ewm(span=days, adjust=False).mean(), range)

    def get_latest_day(self) -> pd.Timestamp:
        return max(self.data.index)