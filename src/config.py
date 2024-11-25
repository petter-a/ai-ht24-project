import pandas as pd
import os

# ====================================================
# Project directory
# ====================================================
# Differences in rootpath between general dev environment 
# and notebooks requires a fixed path
_ = os.path.dirname(os.path.abspath(__file__))

# ====================================================
# Custom types
# ====================================================
type DateRange = tuple[pd.Timestamp | None, pd.Timestamp | None]

# ====================================================
# Paths
# ====================================================
model_path = f'{_}/../models'
tuner_path = f'{_}/../tuning'
cache_path = f'{_}/../data'
