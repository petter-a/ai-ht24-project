from loader import Loader 
from model import StockModel
from plot import Plot

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
    
    for symbol in args.symbols:
        stock = loader.get_stock(symbol)        
        
        # Create and save model
        Plot(stock, ('2024-10-01', '2024-11-01')).draw()
main()

