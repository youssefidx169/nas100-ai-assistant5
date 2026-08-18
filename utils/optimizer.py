"""
Walk-forward parameter optimization for the NAS100 breakout strategy.
Searches over indicator periods, thresholds, and risk parameters
to maximize risk-adjusted returns (Sharpe ratio).
"""

import pandas as pd
import numpy as np
from itertools import product
from utils.support_resistance import detect_support_resistance
from utils.trade_signals import generate_trade_signals
from utils.backtest import backtest_strategy


def optimize_strategy(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    starting_capital: float = 10000.0,
    param_grid: dict = None,
    optimize_metric: str = 'sharpe_ratio',
    top_n: int = 5,
    progress_callback=None,
) -> dict:
    """
    Walk-forward optimization: train on first 70% of data, validate on last 30%.

    Args:
        df: Full OHLCV DataFrame
        train_ratio: Fraction of data used for training
        starting_capital: Initial capital for backtesting
        param_grid: Dict of parameter lists to search over
        optimize_metric: Metric to maximize ('sharpe_ratio', 'total_return_pct',
                        'profit_factor', 'expectancy')
        top_n: Number of top parameter sets to return
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        Dict with keys:
            - 'best_params': Best parameter set
            - 'best_train_metrics': Metrics on training data
            - 'best_test_metrics': Metrics on test (out-of-sample) data
            - 'top_results': List of top_n results
            - 'total_combinations': Total parameter combinations tested
    """
    if param_grid is None:
        param_grid = get_default_param_grid()

    # Split data
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    total = len(combinations)

    results = []

    for idx, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        if progress_callback:
            progress_callback(idx + 1, total)

        try:
            # Extract parameters
            sr_lookback = params.get('sr_lookback', 5)
            sr_tolerance = params.get('sr_tolerance', 0.5)
            ema_fast = params.get('ema_fast', 9)
            ema_slow = params.get('ema_slow', 21)
            rsi_period = params.get('rsi_period', 14)
            sl_atr_mult = params.get('sl_atr_mult', 1.5)
            tp_atr_mult = params.get('tp_atr_mult', 3.0)
            risk_pct = params.get('risk_pct', 1.0)
            vol_threshold = params.get('vol_threshold', 1.5)

            # Validate: ema_fast must be less than ema_slow
            if ema_fast >= ema_slow:
                continue

            indicator_config = {
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'rsi_period': rsi_period,
                'vol_threshold': vol_threshold,
            }

            # --- Train ---
            zones_train = detect_support_resistance(
                df_train, lookback=sr_lookback, tolerance_pct=sr_tolerance
            )

            if not zones_train:
                continue

            signals_train = generate_trade_signals(
                df_train, zones_train, config=indicator_config
            )

            if signals_train.empty:
                continue

            trades_train, equity_train, metrics_train = backtest_strategy(
                df_train, signals_train,
                sl_atr_mult=sl_atr_mult,
                tp_atr_mult=tp_atr_mult,
                risk_pct=risk_pct,
                starting_capital=starting_capital,
                use_trailing_stop=True,
            )

            # Skip if not enough trades
            if metrics_train['total_trades'] < 5:
                continue

            # --- Test (out-of-sample) ---
            zones_test = detect_support_resistance(
                df_test, lookback=sr_lookback, tolerance_pct=sr_tolerance
            )

            if not zones_test:
                continue

            signals_test = generate_trade_signals(
                df_test, zones_test, config=indicator_config
            )

            if signals_test.empty:
                continue

            trades_test, equity_test, metrics_test = backtest_strategy(
                df_test, signals_test,
                sl_atr_mult=sl_atr_mult,
                tp_atr_mult=tp_atr_mult,
                risk_pct=risk_pct,
                starting_capital=starting_capital,
                use_trailing_stop=True,
            )

            # Score by training metric
            train_score = metrics_train.get(optimize_metric, 0)
            test_score = metrics_test.get(optimize_metric, 0)

            # Penalize overfitting: if test score is much worse than train
            if isinstance(train_score, (int, float)) and isinstance(test_score, (int, float)):
                if train_score > 0:
                    robustness = test_score / train_score if train_score != 0 else 0
                else:
                    robustness = 0
            else:
                robustness = 0

            results.append({
                'params': params,
                'train_score': train_score,
                'test_score': test_score,
                'robustness': round(robustness, 2) if isinstance(robustness, float) else 0,
                'train_metrics': metrics_train,
                'test_metrics': metrics_test,
                'train_trades': len(trades_train),
                'test_trades': len(trades_test),
            })

        except Exception:
            continue

    if not results:
        return {
            'best_params': get_default_params(),
            'best_train_metrics': {},
            'best_test_metrics': {},
            'top_results': [],
            'total_combinations': total,
        }

    # Sort by combined score (train_score * robustness) to favor robust results
    results.sort(
        key=lambda x: x['train_score'] * max(x['robustness'], 0.1)
            if isinstance(x['train_score'], (int, float)) else 0,
        reverse=True,
    )

    best = results[0]

    return {
        'best_params': best['params'],
        'best_train_metrics': best['train_metrics'],
        'best_test_metrics': best['test_metrics'],
        'top_results': results[:top_n],
        'total_combinations': total,
    }


def get_default_param_grid() -> dict:
    """
    Default parameter grid for optimization.
    Kept small enough to run in reasonable time (~100-200 combinations).
    """
    return {
        'sr_lookback': [3, 5, 8],
        'sr_tolerance': [0.3, 0.5, 0.8],
        'ema_fast': [8, 9, 12],
        'ema_slow': [21, 26],
        'sl_atr_mult': [1.0, 1.5, 2.0],
        'tp_atr_mult': [2.0, 3.0, 4.0],
        'risk_pct': [1.0, 1.5],
    }


def get_default_params() -> dict:
    """Default parameters (used when optimization finds no results)."""
    return {
        'sr_lookback': 5,
        'sr_tolerance': 0.5,
        'ema_fast': 9,
        'ema_slow': 21,
        'rsi_period': 14,
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 3.0,
        'risk_pct': 1.0,
        'vol_threshold': 1.5,
    }
