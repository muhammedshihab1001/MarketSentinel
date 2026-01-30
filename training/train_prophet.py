import joblib
from app.services.data_fetcher import StockPriceFetcher
from app.models.prophet_model import train_prophet
import datetime

END_DATE = datetime.date.today().isoformat()
fetcher = StockPriceFetcher()
df = fetcher.fetch("AAPL", "2018-01-01", END_DATE)

model = train_prophet(df)

joblib.dump(model, "models/prophet_trend.pkl")
print("Prophet model saved")
