from loader import Loader 
from model import StockModel
from plot import Plot
import pandas as pd

import argparse

def main():
    parser = argparse.ArgumentParser(prog='train')
    parser.add_argument("--refresh", action="store_true", help="Download latest datasets")
    parser.add_argument("--list", action="store_true", help="List all available symbols")
    parser.add_argument('symbols', nargs='*', default=[], help='Stock symbols to build')
    args = parser.parse_args()

    # Load and prepare data
    if args.refresh:
        loader = Loader().create_preprocessed_data()
    else:
        loader = Loader().load_preprocessed_data()

    if args.list:
        print(loader.list_companies())
        exit(0)

    if args.symbols == []:
        parser.print_help()
    
    today = pd.to_datetime("today")
    three_months_ago = today - pd.DateOffset(months=3)

    for symbol in args.symbols:
        stock = loader.get_stock(symbol)        
        
        # Create and save model
        Plot(stock, (three_months_ago, None)).draw()
main()

