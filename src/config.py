import pandas as pd
import os

# Custom types
type DateRange = tuple[pd.Timestamp | None, pd.Timestamp | None]

# Configurables
source_path = os.path.dirname(os.path.abspath(__file__))
models_path = f'{source_path}/../models'
data_path = f'{source_path}/../data'
