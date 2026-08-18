"""
End-to-end test script for the NAS100 strategy pipeline.
Downloads data → detects S/R → generates signals → backtests → reports metrics.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_provider import download_nas100, get_data_summary
from utils.support_resistance import detect_support_resistance
from utils.trade_signals import generate_trade_signals
from utils.backtest import backtest_strategy
from utils.indicators import add_all_indicators


def main():
    print("=" * 60)
    print("NAS100 Strategy Pipeline — End-to-End Test")
    print("=" * 60)

    # 1. Download data
    print("\n[1/5] Downloading NAS100 data (1D, 2 years)...")
    try:
        df = download_nas100(interval="1d", period_days=730)
        summary = get_data_summary(df)
        print(f"  ✅ Downloaded {summary['rows']} bars")
        print(f"  📅 Range: {summary['start']} → {summary['end']}")
        print(f"  💰 Price range: {summary['price_range']}")
        print(f"  📊 Avg volume: {summary['avg_volume']}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return

    # 2. Add indicators
    print("\n[2/5] Computing indicators...")
    try:
        df_ind = add_all_indicators(df)
        indicator_cols = [c for c in df_ind.columns if c not in df.columns]
        print(f"  ✅ Added {len(indicator_cols)} indicators: {', '.join(indicator_cols)}")
        print(f"  📈 Last RSI: {df_ind['RSI'].iloc[-1]:.1f}")
        print(f"  📈 Last ATR: {df_ind['ATR'].iloc[-1]:.1f}")
        print(f"  📈 Last EMA9: {df_ind['EMA_fast'].iloc[-1]:.1f}")
        print(f"  📈 Last EMA21: {df_ind['EMA_slow'].iloc[-1]:.1f}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback; traceback.print_exc()
        return

    # 3. Detect S/R zones
    print("\n[3/5] Detecting support & resistance zones...")
    try:
        zones = detect_support_resistance(df, lookback=5, tolerance_pct=0.5)
        print(f"  ✅ Found {len(zones)} zones")
        for i, z in enumerate(zones[:5]):
            print(f"    Zone {i+1}: {z['type'].title()} @ {z['level']:.0f} "
                  f"(touches={z['touches']}, strength={z['strength']:.0%})")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback; traceback.print_exc()
        return

    # 4. Generate signals
    print("\n[4/5] Generating trade signals...")
    try:
        signals = generate_trade_signals(df, zones, use_volume=True)
        print(f"  ✅ Generated {len(signals)} signals")
        if not signals.empty:
            longs = len(signals[signals['signal'] == 'breakout_long'])
            shorts = len(signals[signals['signal'] == 'breakout_short'])
            print(f"    📈 Long signals: {longs}")
            print(f"    📉 Short signals: {shorts}")
            print(f"  Columns: {list(signals.columns)}")
            print(f"\n  First 3 signals:")
            print(signals.head(3).to_string(index=False))
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback; traceback.print_exc()
        return

    # 5. Backtest
    print("\n[5/5] Running backtest...")
    try:
        trades, equity, metrics = backtest_strategy(
            df, signals,
            sl_atr_mult=1.5,
            tp_atr_mult=3.0,
            risk_pct=1.0,
            starting_capital=10000,
            commission_per_trade=2.0,
            use_trailing_stop=True,
        )
        print(f"  ✅ Backtest complete")
        print(f"\n  {'─' * 40}")
        print(f"  BACKTEST RESULTS")
        print(f"  {'─' * 40}")
        print(f"  Total Trades:      {metrics['total_trades']}")
        print(f"  Win Rate:          {metrics['win_rate']}%")
        print(f"  Total Return:      {metrics['total_return_pct']}%")
        print(f"  Total PnL:         ${metrics['total_pnl']:,.2f}")
        print(f"  Final Equity:      ${metrics['final_equity']:,.2f}")
        print(f"  Profit Factor:     {metrics['profit_factor']}")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']}")
        print(f"  Max Drawdown:      {metrics['max_drawdown_pct']}%")
        print(f"  Avg Win:           ${metrics['avg_win']:,.2f}")
        print(f"  Avg Loss:          ${metrics['avg_loss']:,.2f}")
        print(f"  Expectancy:        ${metrics['expectancy']:,.2f}")
        print(f"  Avg R-Multiple:    {metrics['avg_r_multiple']}")
        print(f"  Consec Wins:       {metrics['max_consecutive_wins']}")
        print(f"  Consec Losses:     {metrics['max_consecutive_losses']}")
        print(f"  Avg Duration:      {metrics['avg_trade_duration']}")
        print(f"  {'─' * 40}")

        if not trades.empty:
            print(f"\n  First 5 trades:")
            print(trades.head(5).to_string(index=False))
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback; traceback.print_exc()
        return

    print(f"\n{'=' * 60}")
    print("✅ ALL PIPELINE STAGES PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
