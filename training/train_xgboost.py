import os
import datetime
import tempfile
import shutil
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier

from core.data.data_fetcher import StockPriceFetcher
from core.data.news_fetcher import NewsFetcher
from core.sentiment.sentiment import SentimentAnalyzer
from core.features.feature_engineering import FeatureEngineer
from core.schema.feature_schema import MODEL_FEATURES, validate_feature_schema
from training.backtesting.walk_forward import WalkForwardValidator


MODEL_PATH = "artifacts/xgboost/model.pkl"

SEED = 42
np.random.seed(SEED)

TRAINING_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"
]

MIN_TRAINING_ROWS = 1500

PROB_THRESHOLD = float(
    os.getenv("SIGNAL_THRESHOLD", "0.58")
)


def save_model_atomic(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path)) as tmp:
        joblib.dump(model, tmp.name)
        temp_name = tmp.name

    shutil.move(temp_name, path)

    if not os.path.exists(path):
        raise RuntimeError("Model write failed.")


def sanitize_dataframe(df):

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        raise RuntimeError("Dataset empty after sanitation.")

    return df


def load_training_data():

    fetcher = StockPriceFetcher()
    news_fetcher = NewsFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    engineer = FeatureEngineer()

    end_date = datetime.date.today().isoformat()

    datasets = []

    for ticker in TRAINING_TICKERS:

        price_df = fetcher.fetch(
            ticker=ticker,
            start_date="2018-01-01",
            end_date=end_date
        )

        news_df = news_fetcher.fetch(
            f"{ticker} stock",
            max_items=100
        )

        scored_df = sentiment_analyzer.analyze_dataframe(news_df)
        sentiment_df = sentiment_analyzer.aggregate_daily_sentiment(scored_df)

        dataset = engineer.build_feature_pipeline(
            price_df,
            sentiment_df,
            training=True
        )

        dataset["ticker"] = ticker
        datasets.append(dataset)

    df = pd.concat(datasets, ignore_index=True)

    df.sort_values(["date", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = sanitize_dataframe(df)

    if len(df) < MIN_TRAINING_ROWS:
        raise RuntimeError(f"Training aborted — dataset too small ({len(df)} rows)")

    validated_features = validate_feature_schema(df)

    df = pd.concat(
        [df[["date", "ticker", "target"]], validated_features],
        axis=1
    )

    return df


def date_split(df):

    cutoff_date = "2023-01-01"

    train = df[df["date"] < cutoff_date]
    val = df[df["date"] >= cutoff_date]

    return train, val


def train_full_model(df):

    train, val = date_split(df)

    X_train = train[list(MODEL_FEATURES)]
    y_train = train["target"]

    X_val = val[list(MODEL_FEATURES)]
    y_val = val["target"]

    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    model = XGBClassifier(
        n_estimators=900,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=SEED,
        tree_method="hist",
        n_jobs=1,
        scale_pos_weight=pos_weight
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    probs = model.predict_proba(X_val)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, probs)),
        "logloss": float(log_loss(y_val, probs)),
    }

    return model
