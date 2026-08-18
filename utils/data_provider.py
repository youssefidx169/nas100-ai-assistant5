"""
NAS100 data provider — download historical data via yfinance.
Supports multiple timeframes and local CSV caching.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# Ticker for NASDAQ-100 index
NAS100_TICKER = "^NDX"

# Cache directory (relative to project root)
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")

# yfinance interval limits (max history for each interval)
INTERVAL_LIMITS = {
    "1m": 7,        # 7 days max
    "2m": 60,
    "5m": 60,       # 60 days max
    "15m": 60,
    "30m": 60,
    "1h": 730,      # ~2 years
    "1d": 10000,    # effectively unlimited
    "1wk": 10000,
    "1mo": 10000,
}


def _cache_path(interval: str, period_days: int) -> str:
    """Generate a cache file path for the given parameters."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"nas100_{interval}_{period_days}d.csv")


def download_nas100(
    interval: str = "1d",
    period_days: int = 365,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download NAS100 (^NDX) historical OHLCV data.

    Args:
        interval: Candle interval — '1m', '5m', '15m', '1h', '1d', '1wk'
        period_days: Number of calendar days of history to fetch
        use_cache: If True, load from local CSV cache if available
        force_refresh: If True, bypass cache and re-download

    Returns:
        DataFrame with columns: datetime, Open, High, Low, Close, Volume
    """
    # Enforce yfinance limits
    max_days = INTERVAL_LIMITS.get(interval, 10000)
    if period_days > max_days:
        period_days = max_days

    cache_file = _cache_path(interval, period_days)

    # Try cache first
    if use_cache and not force_refresh and os.path.exists(cache_file):
        # Check if cache is less than 1 day old for intraday, 1 week for daily
        cache_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
        max_age = 86400 if interval in ("1m", "5m", "15m", "1h") else 604800
        if cache_age < max_age:
            df = pd.read_csv(cache_file)
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df['datetime'] = df['datetime'].dt.tz_localize(None)
            return df

    # Download from yfinance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)

    ticker = yf.Ticker(NAS100_TICKER)
    raw = ticker.history(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for {NAS100_TICKER} with interval={interval}, "
            f"period={period_days}d. Try a shorter period or different interval."
        )

    # Normalize columns
    df = raw.reset_index()

    # yfinance returns 'Date' for daily, 'Datetime' for intraday
    date_col = None
    for col in ['Datetime', 'Date', 'date', 'datetime']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        # Fallback: use index
        df['datetime'] = df.index
    else:
        df.rename(columns={date_col: 'datetime'}, inplace=True)

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    # Strip timezone info to avoid mixed timezone issues
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)

    # Keep only OHLCV columns
    keep_cols = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Drop any rows with NaN prices
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Cache the data
    if use_cache:
        df.to_csv(cache_file, index=False)

    return df


def get_available_intervals() -> list:
    """Return list of supported intervals."""
    return list(INTERVAL_LIMITS.keys())


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary of the downloaded data.
    """
    return {
        "rows": len(df),
        "start": df['datetime'].min().strftime("%Y-%m-%d %H:%M"),
        "end": df['datetime'].max().strftime("%Y-%m-%d %H:%M"),
        "price_range": f"{df['Low'].min():.2f} — {df['High'].max():.2f}",
        "avg_volume": f"{df['Volume'].mean():,.0f}",
    }
