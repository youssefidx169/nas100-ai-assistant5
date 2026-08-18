"""
Trade signal generation with multi-indicator confirmation.
Breakout signals use a scoring system: each confirmation adds to the signal
strength. A minimum score threshold determines if the signal fires.
"""

import pandas as pd
import numpy as np
from utils.indicators import add_all_indicators


def generate_trade_signals(
    df: pd.DataFrame,
    zones: list,
    config: dict = None,
    use_volume: bool = True,
    session_start=None,
    session_end=None,
) -> pd.DataFrame:
    """
    Generate breakout trade signals with multi-indicator scoring.

    Each confirmation criterion adds points:
        - Breakout of S/R zone:     +3 (required)
        - EMA trend alignment:     +2
        - RSI in valid range:      +1
        - Volume spike:            +1
        - Zone strength > 50%:     +1

    A minimum score of 3 is needed (breakout alone is sufficient,
    but higher-scoring signals are higher confidence).

    Args:
        df: OHLCV DataFrame with 'datetime' column
        zones: List of S/R zone dicts from detect_support_resistance()
        config: Optional indicator config overrides
        use_volume: Whether to include volume in scoring
        session_start: Only generate signals after this time
        session_end: Only generate signals before this time

    Returns:
        DataFrame with columns: datetime, price, signal, zone_level,
            zone_strength, atr, rsi, ema_trend, score
    """
    if config is None:
        config = {}

    rsi_long_min = config.get('rsi_long_min', 35)
    rsi_long_max = config.get('rsi_long_max', 80)
    rsi_short_min = config.get('rsi_short_min', 20)
    rsi_short_max = config.get('rsi_short_max', 65)
    vol_threshold = config.get('vol_threshold', 1.3)
    min_score = config.get('min_score', 3)

    # Add indicators
    df_ind = add_all_indicators(df.copy(), config)

    signals = []

    for i in range(1, len(df_ind)):
        row = df_ind.iloc[i]
        prev_row = df_ind.iloc[i - 1]

        # Skip if indicators not ready (need at least 26 bars for MACD)
        if pd.isna(row.get('RSI')) or pd.isna(row.get('ATR')):
            continue

        # Session filter
        if 'datetime' in df_ind.columns:
            dt = pd.to_datetime(row['datetime'])
            if session_start and hasattr(dt, 'time') and dt.time() < session_start:
                continue
            if session_end and hasattr(dt, 'time') and dt.time() > session_end:
                continue

        close = row['Close']
        prev_close = prev_row['Close']
        ema_fast = row.get('EMA_fast', close)
        ema_slow = row.get('EMA_slow', close)
        rsi = row['RSI']
        atr = row['ATR']
        vol_ratio = row.get('Vol_ratio', 1.0)
        macd_hist = row.get('MACD_hist', 0)

        bar_signal_found = False

        for zone in zones:
            if bar_signal_found:
                break

            zone_level = zone['level']
            zone_upper = zone['resistance']
            zone_lower = zone['support']
            zone_strength = zone['strength']

            # ─── LONG BREAKOUT: price crosses above a zone ───
            if prev_close <= zone_upper and close > zone_upper:
                score = 3  # Base breakout score

                # EMA trend alignment
                ema_aligned = False
                if not pd.isna(ema_fast) and not pd.isna(ema_slow):
                    if ema_fast > ema_slow:
                        score += 2
                        ema_aligned = True

                # RSI filter
                if rsi_long_min <= rsi <= rsi_long_max:
                    score += 1

                # Volume spike
                if use_volume and not pd.isna(vol_ratio) and vol_ratio >= vol_threshold:
                    score += 1

                # Zone strength bonus
                if zone_strength >= 0.5:
                    score += 1

                # MACD histogram positive
                if not pd.isna(macd_hist) and macd_hist > 0:
                    score += 1

                if score >= min_score:
                    signals.append({
                        'datetime': row.get('datetime', df_ind.index[i]),
                        'price': close,
                        'signal': 'breakout_long',
                        'zone_level': round(zone_level, 2),
                        'zone_strength': round(zone_strength, 2),
                        'atr': round(atr, 2),
                        'rsi': round(rsi, 1),
                        'ema_trend': 'bullish' if ema_aligned else 'neutral',
                        'score': score,
                    })
                    bar_signal_found = True

            # ─── SHORT BREAKOUT: price crosses below a zone ───
            elif prev_close >= zone_lower and close < zone_lower:
                score = 3  # Base breakout score

                # EMA trend alignment
                ema_aligned = False
                if not pd.isna(ema_fast) and not pd.isna(ema_slow):
                    if ema_fast < ema_slow:
                        score += 2
                        ema_aligned = True

                # RSI filter
                if rsi_short_min <= rsi <= rsi_short_max:
                    score += 1

                # Volume spike
                if use_volume and not pd.isna(vol_ratio) and vol_ratio >= vol_threshold:
                    score += 1

                # Zone strength bonus
                if zone_strength >= 0.5:
                    score += 1

                # MACD histogram negative
                if not pd.isna(macd_hist) and macd_hist < 0:
                    score += 1

                if score >= min_score:
                    signals.append({
                        'datetime': row.get('datetime', df_ind.index[i]),
                        'price': close,
                        'signal': 'breakout_short',
                        'zone_level': round(zone_level, 2),
                        'zone_strength': round(zone_strength, 2),
                        'atr': round(atr, 2),
                        'rsi': round(rsi, 1),
                        'ema_trend': 'bearish' if ema_aligned else 'neutral',
                        'score': score,
                    })
                    bar_signal_found = True

    result = pd.DataFrame(signals)

    # Deduplicate: no two signals within min_gap bars of each other
    if not result.empty:
        result['datetime'] = pd.to_datetime(result['datetime'])
        result = result.sort_values('datetime').reset_index(drop=True)
        result = _deduplicate_signals(result, df_ind, min_gap_bars=3)

    return result


def _deduplicate_signals(
    signals: pd.DataFrame, df: pd.DataFrame, min_gap_bars: int = 3
) -> pd.DataFrame:
    """Remove signals that are too close together (within min_gap_bars)."""
    if signals.empty or len(signals) <= 1:
        return signals

    filtered = [signals.iloc[0].to_dict()]

    for i in range(1, len(signals)):
        prev_time = pd.to_datetime(filtered[-1]['datetime'])
        curr_time = pd.to_datetime(signals.iloc[i]['datetime'])

        # Approximate bar gap using the data index
        if 'datetime' in df.columns:
            prev_mask = df['datetime'] <= prev_time
            curr_mask = df['datetime'] <= curr_time
            prev_idx = df[prev_mask].index
            curr_idx = df[curr_mask].index
            if len(prev_idx) > 0 and len(curr_idx) > 0:
                gap = curr_idx[-1] - prev_idx[-1]
                if gap >= min_gap_bars:
                    filtered.append(signals.iloc[i].to_dict())
            else:
                filtered.append(signals.iloc[i].to_dict())
        else:
            filtered.append(signals.iloc[i].to_dict())

    return pd.DataFrame(filtered).reset_index(drop=True)
