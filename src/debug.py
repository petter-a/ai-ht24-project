import numpy as np
import pandas as pd
from loader import Loader 
from model import StockModel

stock = 'PFE'
loader = Loader()
model = StockModel(loader.get_stock_data(stock))
model.train_model()
model.toFile()


