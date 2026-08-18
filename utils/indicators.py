"""
Centralized technical indicator calculations for the NAS100 strategy.
All functions operate on pandas Series/DataFrames and return pandas objects.
"""

import numpy as np
import pandas as pd


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    Expects columns: 'High', 'Low', 'Close'.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price (cumulative intraday).
    Expects columns: 'High', 'Low', 'Close', 'Volume'.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    cumulative_tp_vol = (typical_price * df['Volume']).cumsum()
    cumulative_vol = df['Volume'].cumsum()
    vwap = cumulative_tp_vol / cumulative_vol
    return vwap


def calc_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands.
    Returns: (upper_band, middle_band, lower_band)
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD (Moving Average Convergence Divergence).
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """
    Stochastic Oscillator.
    Returns: (%K, %D)
    """
    lowest_low = df['Low'].rolling(window=k_period).min()
    highest_high = df['High'].rolling(window=k_period).max()
    k = 100.0 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


def add_all_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame in-place.
    
    Args:
        df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
        config: Optional dict to override default indicator periods
    
    Returns:
        DataFrame with indicator columns added
    """
    if config is None:
        config = {}

    ema_fast = config.get('ema_fast', 9)
    ema_slow = config.get('ema_slow', 21)
    rsi_period = config.get('rsi_period', 14)
    atr_period = config.get('atr_period', 14)
    bb_period = config.get('bb_period', 20)
    bb_std = config.get('bb_std', 2.0)

    df = df.copy()

    # EMAs
    df['EMA_fast'] = calc_ema(df['Close'], ema_fast)
    df['EMA_slow'] = calc_ema(df['Close'], ema_slow)

    # RSI
    df['RSI'] = calc_rsi(df['Close'], rsi_period)

    # ATR
    df['ATR'] = calc_atr(df, atr_period)

    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = calc_bollinger(
        df['Close'], bb_period, bb_std
    )

    # MACD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = calc_macd(df['Close'])

    # Volume SMA (for volume spike detection)
    df['Vol_SMA_20'] = calc_sma(df['Volume'], 20)
    df['Vol_ratio'] = df['Volume'] / df['Vol_SMA_20']

    # VWAP
    if df['Volume'].sum() > 0:
        df['VWAP'] = calc_vwap(df)

    return df
