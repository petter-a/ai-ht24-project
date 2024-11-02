import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stock import Stock
from lib_types import DateRange
import matplotlib.dates as mdates

class Plot:
    def __init__(self, stock: Stock, range: DateRange):
        self.stock = stock
        self.range = range

        (self.fig, self.axs) = plt.subplots(2, 1, figsize=(20, 10), layout='constrained')
        self.axs[0].grid(True)
        self.axs[0].set_title(self.format_title())
        self.fig.suptitle(f'{stock.get_symbol_name()} - {stock.get_company_name()}', size=20)        

        self.axs[0].xaxis.set_minor_locator(mdates.DayLocator())
        self.axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(self.axs[0].xaxis.get_major_locator()))

    def format_title(self):
        return f'Date: {self.stock.get_closing_date().strftime('%Y-%m-%d')}\nClosing price: ${self.stock.get_closing_price():1.2f}'

    def draw(self):
        self.plot_stock_price()
        self.plot_stock_prediction()
        self.plot_sma200()
        self.plot_sma50()
        self.plot_mean_price()
        self.plot_max_price()
        self.plot_min_price()
        self.plot_volume()
        self.axs[0].legend()


    def plot_stock_price(self):
        prices = self.stock.get_price(self.range)
        self.axs[0].plot(prices, color='green', label='Price')

    def plot_stock_prediction(self):
        prices = self.stock.get_predicted_price()['Adj Close']
        self.axs[0].plot(prices, color='grey', label='Predicted Price')

    def plot_max_price(self):
        prices = self.stock.get_price(self.range)
        self.axs[0].axhline(y=prices.max(), color="grey", linestyle="--", label=prices.max())
        #self.ax.plot(prices.filter([prices.idxmax()]), marker='*')

    def plot_min_price(self):
        prices = self.stock.get_price(self.range)
        self.axs[0].axhline(y=prices.min(), color="grey", linestyle="--", label=prices.min())

    def plot_sma200(self):
        prices = self.stock.get_sma(200, self.range)
        self.axs[0].plot(prices, color='red', label='SMA200')
        
    def plot_sma50(self):
        prices = self.stock.get_sma(50, self.range)
        self.axs[0].plot(prices, color='blue', label='SMA50')

    def plot_mean_price(self):
        prices = self.stock.get_price(self.range)
        self.axs[0].axhline(y=prices.mean(), color='brown', label=prices.mean())

    def plot_volume(self):
        volume = self.stock.get_volume(self.range)
        self.axs[1].bar(volume.index, volume)
    
    def show(self):
        plt.show()
