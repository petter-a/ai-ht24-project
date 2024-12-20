from loader import Loader 
from model import StockModel
from plot import Plot
import pandas as pd

import argparse

def main():
    parser = argparse.ArgumentParser(prog='predict.py')
    parser.add_argument("--fromcache", action="store_true", help="Use pre-processed data (if available)")
    parser.add_argument("--list", action="store_true", help="List all available symbols")
    parser.add_argument('symbols', nargs='*', default=[], help='Stock symbols to predict')
    args = parser.parse_args()

    # ====================================================
    # Load and prepare data
    # ====================================================
    if args.fromcache:
        loader = Loader().load_preprocessed_data()
    else:
        loader = Loader().create_preprocessed_data()

    # ====================================================
    # List companies
    # ====================================================
    if args.list:
        print(loader.list_companies().to_string(columns=['Shortname']))
        exit(0)

    # ====================================================
    # Missing arguments
    # ====================================================
    if args.symbols == []:
        parser.print_help()
    
    # ====================================================
    # Perform prediction
    # ====================================================
    today = pd.to_datetime("today")
    three_months_ago = today - pd.DateOffset(months=3)

    for symbol in args.symbols:
        if not loader.is_valid_symbol(symbol):
            print(f'Unknown symbol {symbol}')
            continue

        stock = loader.get_stock(symbol)        
        
        if not stock.has_data():
            print(f'Data for stock symbol {stock.get_symbol_name()} is missing')
            continue

        # Plot prediction
        Plot(stock, (three_months_ago, None)).draw()
main()

