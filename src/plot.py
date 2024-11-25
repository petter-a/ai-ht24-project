import matplotlib.pyplot as plt
from stock import Stock
from config import DateRange
import matplotlib.dates as mdates

class Plot:
    def __init__(self, stock: Stock, range: DateRange):
        self.stock = stock
        self.range = range

        (self.fig, self.axs) = plt.subplots(3, 1, figsize=(20, 10), layout='constrained')
        self.axs[0].grid(True)
        self.axs[0].set_title(self.format_title())
        self.fig.suptitle(f'{stock.get_symbol_name()} - {stock.get_company_name()}', size=20)        

        self.axs[0].xaxis.set_minor_locator(mdates.DayLocator())
        self.axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(self.axs[0].xaxis.get_major_locator()))
        self.axs[0].yaxis.set_major_formatter('${x:1.2f}')

    def format_title(self):
        return f'Date: {self.stock.get_closing_date().strftime('%Y-%m-%d')}\nClosing price: ${self.stock.get_closing_price():1.2f}'

    def draw(self):
        self.plot_stock_price()
        self.plot_volume()
        self.plot_sma200()
        self.plot_sma50()
        self.plot_mean_price()
        self.plot_max_price()
        self.plot_min_price()
        self.plot_rsi()
        self.plot_stock_prediction()
        self.axs[0].legend()
        plt.show(block=True)


    def plot_stock_price(self):
        prices = self.stock.get_price(self.range)
        self.axs[0].plot(prices, color='green', label='Price', linewidth=3)

    def plot_stock_prediction(self):
        prices = self.stock.get_predicted_price()
        self.axs[0].axvline(prices.index[0])
        self.axs[0].plot(prices['Close'], color='green', label='Predicted Price', linestyle=(0,(2,2)), linewidth=3)
        (bottom, top) = self.axs[0].get_ylim()
        self.axs[0].text(prices.index[0], top - ((top - bottom)/2), 'Future', rotation=90, verticalalignment='center')
        self.axs[0].plot(prices['SMA_high'], color='red', label='SMA 200', linestyle="dotted", alpha=0.2)
        self.axs[0].plot(prices['SMA_low'], color='blue', label='SMA 50', linestyle="dotted", alpha=0.2)
        self.axs[2].axvline(prices.index[0])
        self.axs[2].bar(prices.index, prices['Volume'], color='grey', label='Predicted Volume')
        (bottom, top) = self.axs[2].get_ylim()
        self.axs[2].text(prices.index[0], top - ((top - bottom)/2), 'Future', rotation=90, verticalalignment='center')

    def plot_max_price(self):
        prices = self.stock.get_price(self.range)
        self.axs[0].axhline(y=prices.max(), color="grey", linestyle="--", label=f'Max: ${prices.max():1.2f}', alpha=0.2)
        #self.ax.plot(prices.filter([prices.idxmax()]), marker='*')

    def plot_min_price(self):
        prices = self.stock.get_price(self.range)
        self.axs[0].axhline(y=prices.min(), color="grey", linestyle="--", label=f'Min: ${prices.min():1.2f}', alpha=0.2)

    def plot_sma200(self):
        prices = self.stock.get_sma_high(self.range)
        self.axs[0].plot(prices, color='red', label='SMA200', alpha=0.3)
        
    def plot_sma50(self):
        prices = self.stock.get_sma_low(self.range)
        self.axs[0].plot(prices, color='blue', label='SMA50', alpha=0.3)

    def plot_mean_price(self):
        prices = self.stock.get_price(self.range)
        self.axs[0].axhline(y=prices.mean(), color='brown', linestyle="--", label=f'Medel: ${prices.mean():1.2f}', alpha=0.2)

    def plot_rsi(self):
        prices = self.stock.get_rsi_val(self.range)
        self.axs[1].set_title('RSI - Relative Strength Index')
        self.axs[1].axhline(y=[70], color='grey', linestyle="--")
        self.axs[1].plot(prices, color='blue', label='RSI')
        self.axs[1].axhline(y=[30], color='grey', linestyle="--")

    def plot_volume(self):
        volume = self.stock.get_volume(self.range)
        self.axs[2].bar(volume.index, volume, label="Volume")
