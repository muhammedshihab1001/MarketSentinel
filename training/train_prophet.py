import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
from core.data.data_fetcher import StockPriceFetcher
from models.prophet_model import train_prophet
import datetime


END_DATE = datetime.date.today().isoformat()
fetcher = StockPriceFetcher()
df = fetcher.fetch("AAPL", "2018-01-01", END_DATE)

model = train_prophet(df)

joblib.dump(model, "artifacts/prophet_trend.pkl")
print("Prophet model saved")
