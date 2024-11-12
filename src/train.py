from loader import Loader 
from model import StockModel
import argparse

def main():
    parser = argparse.ArgumentParser(prog='train')
    parser.add_argument("--refresh", action="store_true", help="Download latest datasets")
    parser.add_argument("--list", action="store_true", help="List all available symbols")
    parser.add_argument("--interactive", action="store_true", help="Display stats as dashboard")
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
        
        print(f'Training {stock.get_company_name()} ({stock.get_symbol_name()})')
        # Create and save model
        model = StockModel(stock.get_data(), stock.get_symbol_name())
        model.train_model(interactive=args.interactive).save_model()
main()

