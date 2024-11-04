from loader import Loader 
from model import StockModel

def main():
    # Load and prepare data
    loader = Loader()

    for stock in loader.get_symbols():
        print(stock.get_company_name())
        # Create and save model
        model = StockModel(stock.get_data())
        model.train_model(interactive=True)
        model.toFile()

main()

