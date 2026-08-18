"""
Support & Resistance detection using pivot-point clustering.
Replaces the naive binning approach with actual price action analysis.
"""

import numpy as np
import pandas as pd


def _find_pivot_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Find swing high pivots: a bar whose High is the highest
    in a window of [lookback] bars on each side.
    """
    highs = df['High'].values
    n = len(highs)
    pivots = pd.Series(np.nan, index=df.index)

    for i in range(lookback, n - lookback):
        window = highs[i - lookback: i + lookback + 1]
        if highs[i] == window.max():
            pivots.iloc[i] = highs[i]

    return pivots


def _find_pivot_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Find swing low pivots: a bar whose Low is the lowest
    in a window of [lookback] bars on each side.
    """
    lows = df['Low'].values
    n = len(lows)
    pivots = pd.Series(np.nan, index=df.index)

    for i in range(lookback, n - lookback):
        window = lows[i - lookback: i + lookback + 1]
        if lows[i] == window.min():
            pivots.iloc[i] = lows[i]

    return pivots


def _cluster_levels(levels: np.ndarray, tolerance_pct: float = 0.5) -> list:
    """
    Cluster nearby price levels together.
    Levels within tolerance_pct % of each other are grouped.

    Returns list of dicts: {'level': mean_price, 'touches': count}
    """
    if len(levels) == 0:
        return []

    levels = np.sort(levels)
    clusters = []
    current_cluster = [levels[0]]

    for i in range(1, len(levels)):
        # If this level is within tolerance of the cluster mean, group it
        cluster_mean = np.mean(current_cluster)
        if abs(levels[i] - cluster_mean) / cluster_mean * 100 <= tolerance_pct:
            current_cluster.append(levels[i])
        else:
            clusters.append({
                'level': np.mean(current_cluster),
                'touches': len(current_cluster),
            })
            current_cluster = [levels[i]]

    # Don't forget the last cluster
    clusters.append({
        'level': np.mean(current_cluster),
        'touches': len(current_cluster),
    })

    return clusters


def detect_support_resistance(
    df: pd.DataFrame,
    lookback: int = 5,
    tolerance_pct: float = 0.5,
    min_touches: int = 2,
) -> list:
    """
    Detect support and resistance zones using pivot-point clustering.

    Args:
        df: OHLCV DataFrame with columns: High, Low, Close
        lookback: Number of bars on each side to confirm a pivot
        tolerance_pct: Percentage threshold for clustering nearby levels
        min_touches: Minimum number of touches for a level to be valid

    Returns:
        List of dicts with keys:
            - 'support': lower boundary of the zone
            - 'resistance': upper boundary of the zone
            - 'level': center price of the zone
            - 'type': 'support' or 'resistance'
            - 'touches': number of times price touched this zone
            - 'strength': normalized strength score (0-1)
    """
    # Find pivots
    pivot_highs = _find_pivot_highs(df, lookback).dropna().values
    pivot_lows = _find_pivot_lows(df, lookback).dropna().values

    # Cluster resistance levels (from swing highs)
    resistance_clusters = _cluster_levels(pivot_highs, tolerance_pct)
    resistance_clusters = [c for c in resistance_clusters if c['touches'] >= min_touches]

    # Cluster support levels (from swing lows)
    support_clusters = _cluster_levels(pivot_lows, tolerance_pct)
    support_clusters = [c for c in support_clusters if c['touches'] >= min_touches]

    # Current price for context
    current_price = df['Close'].iloc[-1]
    price_range = df['High'].max() - df['Low'].min()
    zone_width = price_range * 0.008  # 0.8% of total range for zone width

    zones = []

    # Build resistance zones
    max_touches = max(
        [c['touches'] for c in resistance_clusters + support_clusters] or [1]
    )

    for cluster in resistance_clusters:
        level = cluster['level']
        zones.append({
            'support': level - zone_width,
            'resistance': level + zone_width,
            'level': level,
            'type': 'resistance',
            'touches': cluster['touches'],
            'strength': cluster['touches'] / max_touches,
        })

    for cluster in support_clusters:
        level = cluster['level']
        zones.append({
            'support': level - zone_width,
            'resistance': level + zone_width,
            'level': level,
            'type': 'support',
            'touches': cluster['touches'],
            'strength': cluster['touches'] / max_touches,
        })

    # Sort by strength (strongest first)
    zones.sort(key=lambda z: z['strength'], reverse=True)

    return zones
