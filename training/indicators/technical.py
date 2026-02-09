import pandas as pd


def moving_average(df, window=20):
    return df["Close"].rolling(window=window).mean()


def rsi(df, window=14):
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(df, window=20):
    ma = moving_average(df, window)
    std = df["Close"].rolling(window).std()

    upper = ma + (2 * std)
    lower = ma - (2 * std)

    return upper, lower


def macd(df):
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()

    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()

    return macd_line, signal
