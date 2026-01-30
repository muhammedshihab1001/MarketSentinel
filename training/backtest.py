import numpy as np
import pandas as pd


def backtest_strategy(df, signal_column="signal"):
    df = df.copy()
    df["return"] = df["close"].pct_change()

    df["strategy_return"] = np.where(
        df[signal_column] == "BUY",
        df["return"],
        np.where(df[signal_column] == "SELL", -df["return"], 0)
    )

    df["cumulative_market"] = (1 + df["return"]).cumprod()
    df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()

    total_return = df["cumulative_strategy"].iloc[-1] - 1
    max_drawdown = (df["cumulative_strategy"].cummax() - df["cumulative_strategy"]).max()

    sharpe = (
        df["strategy_return"].mean()
        / df["strategy_return"].std()
        * np.sqrt(252)
        if df["strategy_return"].std() != 0 else 0
    )

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe)
    }

def signal_hit_rate(df, signal_column="signal"):
    df = df.copy()
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1

    hits = df[
        ((df[signal_column] == "BUY") & (df["future_return"] > 0)) |
        ((df[signal_column] == "SELL") & (df["future_return"] < 0))
    ]

    total_signals = (df[signal_column].isin(["BUY", "SELL"])).sum()

    return {
        "hit_rate": len(hits) / total_signals if total_signals > 0 else 0
    }
