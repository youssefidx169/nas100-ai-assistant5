"""
Professional backtesting engine with position sizing, dynamic SL/TP,
trailing stops, transaction costs, and comprehensive performance metrics.
"""

import pandas as pd
import numpy as np


def backtest_strategy(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 3.0,
    risk_pct: float = 1.0,
    starting_capital: float = 10000.0,
    commission_per_trade: float = 2.0,
    use_trailing_stop: bool = True,
    trailing_atr_mult: float = 2.0,
    max_open_positions: int = 1,
) -> tuple:
    """
    Run a backtest on the generated signals.

    Args:
        df: Full OHLCV DataFrame with 'datetime' column
        signals: Signals DataFrame with columns: datetime, price, signal, atr
        sl_atr_mult: Stop loss distance as multiple of ATR
        tp_atr_mult: Take profit distance as multiple of ATR
        risk_pct: Percentage of equity to risk per trade
        starting_capital: Initial account balance
        commission_per_trade: Fixed commission per round-trip trade
        use_trailing_stop: Whether to use ATR-based trailing stop
        trailing_atr_mult: Trailing stop distance as multiple of ATR
        max_open_positions: Maximum concurrent positions (1 = no overlap)

    Returns:
        (trades_df, equity_curve, metrics_dict)
    """
    trades = []
    equity = [starting_capital]
    balance = starting_capital
    open_positions = 0

    for _, signal in signals.iterrows():
        if open_positions >= max_open_positions:
            continue

        entry_price = signal['price']
        entry_time = signal['datetime']
        direction = signal['signal']
        atr = signal.get('atr', entry_price * 0.01)  # Fallback: 1% of price

        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.01

        # Dynamic SL/TP based on ATR
        sl_distance = atr * sl_atr_mult
        tp_distance = atr * tp_atr_mult

        if direction == 'breakout_long':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        elif direction == 'breakout_short':
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        else:
            continue

        # Position sizing: risk_pct of equity
        risk_amount = balance * (risk_pct / 100.0)
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0

        if position_size <= 0:
            continue

        # Simulate the trade bar-by-bar
        df_slice = df[df['datetime'] > entry_time].copy()
        exit_price = None
        exit_time = None
        trailing_sl = sl_price

        for _, row in df_slice.iterrows():
            high = row['High']
            low = row['Low']

            if direction == 'breakout_long':
                # Update trailing stop
                if use_trailing_stop:
                    new_trail = high - (atr * trailing_atr_mult)
                    if new_trail > trailing_sl:
                        trailing_sl = new_trail

                # Check stop loss (trailing or fixed)
                effective_sl = trailing_sl if use_trailing_stop else sl_price
                if low <= effective_sl:
                    exit_price = effective_sl
                    exit_time = row['datetime']
                    break
                # Check take profit
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_time = row['datetime']
                    break

            elif direction == 'breakout_short':
                # Update trailing stop
                if use_trailing_stop:
                    new_trail = low + (atr * trailing_atr_mult)
                    if new_trail < trailing_sl:
                        trailing_sl = new_trail

                # Check stop loss (trailing or fixed)
                effective_sl = trailing_sl if use_trailing_stop else sl_price
                if high >= effective_sl:
                    exit_price = effective_sl
                    exit_time = row['datetime']
                    break
                # Check take profit
                elif low <= tp_price:
                    exit_price = tp_price
                    exit_time = row['datetime']
                    break

        # If trade never hit SL or TP, close at last available price
        if exit_price is None and len(df_slice) > 0:
            exit_price = df_slice.iloc[-1]['Close']
            exit_time = df_slice.iloc[-1]['datetime']

        if exit_price is not None:
            # Calculate PnL
            if direction == 'breakout_long':
                pnl_per_unit = exit_price - entry_price
            else:
                pnl_per_unit = entry_price - exit_price

            gross_pnl = pnl_per_unit * position_size
            net_pnl = gross_pnl - commission_per_trade
            result = 'Win' if net_pnl > 0 else 'Loss'

            # R-multiple (how many R's did we make/lose)
            r_multiple = pnl_per_unit / sl_distance if sl_distance > 0 else 0

            trades.append({
                'entry_time': entry_time,
                'entry_price': round(entry_price, 2),
                'exit_time': exit_time,
                'exit_price': round(exit_price, 2),
                'direction': direction,
                'sl_price': round(sl_price, 2),
                'tp_price': round(tp_price, 2),
                'position_size': round(position_size, 4),
                'gross_pnl': round(gross_pnl, 2),
                'commission': commission_per_trade,
                'net_pnl': round(net_pnl, 2),
                'r_multiple': round(r_multiple, 2),
                'Result': result,
            })

            balance += net_pnl
            equity.append(round(balance, 2))

    trades_df = pd.DataFrame(trades)
    metrics = calculate_metrics(trades_df, equity, starting_capital)

    return trades_df, equity, metrics


def calculate_metrics(trades_df: pd.DataFrame, equity: list, starting_capital: float) -> dict:
    """
    Calculate comprehensive backtest performance metrics.
    """
    metrics = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'total_return_pct': 0.0,
        'total_pnl': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'largest_win': 0.0,
        'largest_loss': 0.0,
        'profit_factor': 0.0,
        'avg_r_multiple': 0.0,
        'expectancy': 0.0,
        'max_drawdown_pct': 0.0,
        'max_drawdown_value': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'calmar_ratio': 0.0,
        'avg_trade_duration': 'N/A',
        'max_consecutive_wins': 0,
        'max_consecutive_losses': 0,
        'final_equity': starting_capital,
    }

    if trades_df.empty:
        return metrics

    n = len(trades_df)
    wins = trades_df[trades_df['Result'] == 'Win']
    losses = trades_df[trades_df['Result'] == 'Loss']

    metrics['total_trades'] = n
    metrics['winning_trades'] = len(wins)
    metrics['losing_trades'] = len(losses)
    metrics['win_rate'] = round(len(wins) / n * 100, 2) if n > 0 else 0

    # PnL stats
    metrics['total_pnl'] = round(trades_df['net_pnl'].sum(), 2)
    metrics['final_equity'] = round(equity[-1], 2)
    metrics['total_return_pct'] = round(
        (equity[-1] - starting_capital) / starting_capital * 100, 2
    )

    if len(wins) > 0:
        metrics['avg_win'] = round(wins['net_pnl'].mean(), 2)
        metrics['largest_win'] = round(wins['net_pnl'].max(), 2)

    if len(losses) > 0:
        metrics['avg_loss'] = round(losses['net_pnl'].mean(), 2)
        metrics['largest_loss'] = round(losses['net_pnl'].min(), 2)

    # Profit Factor
    gross_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0
    metrics['profit_factor'] = round(
        gross_profit / gross_loss if gross_loss > 0 else float('inf'), 2
    )

    # R-multiples
    metrics['avg_r_multiple'] = round(trades_df['r_multiple'].mean(), 2)

    # Expectancy
    if n > 0:
        win_rate_dec = len(wins) / n
        avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
        avg_loss_abs = abs(losses['net_pnl'].mean()) if len(losses) > 0 else 0
        metrics['expectancy'] = round(
            (win_rate_dec * avg_win) - ((1 - win_rate_dec) * avg_loss_abs), 2
        )

    # Drawdown
    equity_series = pd.Series(equity)
    rolling_max = equity_series.cummax()
    drawdown = equity_series - rolling_max
    metrics['max_drawdown_value'] = round(drawdown.min(), 2)
    metrics['max_drawdown_pct'] = round(
        (drawdown / rolling_max).min() * 100, 2
    ) if rolling_max.max() > 0 else 0

    # Sharpe Ratio (annualized, assuming ~252 trading days)
    if n > 1:
        returns = pd.Series(equity).pct_change().dropna()
        if returns.std() > 0:
            metrics['sharpe_ratio'] = round(
                (returns.mean() / returns.std()) * np.sqrt(252), 2
            )

        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics['sortino_ratio'] = round(
                (returns.mean() / downside_returns.std()) * np.sqrt(252), 2
            )

    # Calmar Ratio
    annual_return = metrics['total_return_pct']
    if abs(metrics['max_drawdown_pct']) > 0:
        metrics['calmar_ratio'] = round(
            annual_return / abs(metrics['max_drawdown_pct']), 2
        )

    # Consecutive wins/losses
    if n > 0:
        results = trades_df['Result'].values
        max_wins = max_losses = current_wins = current_losses = 0
        for r in results:
            if r == 'Win':
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        metrics['max_consecutive_wins'] = max_wins
        metrics['max_consecutive_losses'] = max_losses

    # Average trade duration
    if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
        try:
            durations = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
            avg_dur = durations.mean()
            metrics['avg_trade_duration'] = str(avg_dur).split('.')[0]  # Remove microseconds
        except Exception:
            pass

    return metrics


def format_metrics_display(metrics: dict) -> dict:
    """Format metrics for display in the dashboard."""
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'pct' in key or 'rate' in key or 'return' in key:
                formatted[key] = f"{value}%"
            elif 'ratio' in key or 'factor' in key:
                formatted[key] = f"{value:.2f}"
            else:
                formatted[key] = f"${value:,.2f}" if abs(value) > 10 else f"{value:.2f}"
        else:
            formatted[key] = str(value)
    return formatted
