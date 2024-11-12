import numpy as np
import pandas as pd
from loader import Loader 
from skmodel import StockModel
from plot import Plot

name = 'NVDA'
loader = Loader()
loader.create_preprocessed_data()
loader.load_preprocessed_data()
data = loader.get_stock_data()
stock = loader.get_stock(name)
model = StockModel(data[data['Symbol'].isin(["PFE"])], 'test')
model.train_model()
model.save_model()
Plot(stock, ("2024-08-12", None)).draw()

#print(a)

