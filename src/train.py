from loader import Loader 
from model import StockModel
import argparse

def main():
    # ====================================================
    # Parse commandline arguments
    # ====================================================
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument("--fromcache", action="store_true", help="Use pre-processed data (if available)")
    parser.add_argument("--force_tuner", action="store_true", help="Force tuning of training parameters")
    parser.add_argument("--list", action="store_true", help="List all available symbols")
    parser.add_argument("--interactive", action="store_true", help="Display stats as dashboard")
    parser.add_argument('symbols', nargs='*', default=[], help='Stock symbols to build')
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
    # Perform training
    # ====================================================
    for symbol in args.symbols:
        if not loader.is_valid_symbol(symbol):
            print(f'Unknown symbol {symbol}')
            continue

        stock = loader.get_stock(symbol)        
        
        if not stock.has_data():
            print(f'Data for stock symbol {stock.get_symbol_name()} is missing')
            continue

        print(f'Training {stock.get_company_name()} ({stock.get_symbol_name()})')
        # Create and save model
        model = StockModel(stock.get_data(), stock.get_symbol_name())
        model.train_model(interactive=args.interactive, force_tuner=args.force_tuner).save_model()
main()

