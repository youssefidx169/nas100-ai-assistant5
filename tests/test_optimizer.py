"""
Run walk-forward optimization to find the best parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_provider import download_nas100
from utils.optimizer import optimize_strategy


def main():
    print("=" * 60)
    print("NAS100 Walk-Forward Optimization")
    print("=" * 60)

    # Download data
    print("\n📡 Downloading NAS100 data...")
    df = download_nas100(interval="1d", period_days=730)
    print(f"  ✅ {len(df)} bars loaded")

    # Run optimization
    print("\n🔄 Running optimization (this may take a minute)...")

    def progress(current, total):
        if current % 50 == 0 or current == total:
            print(f"  [{current}/{total}] {current/total*100:.0f}%")

    result = optimize_strategy(
        df,
        train_ratio=0.7,
        starting_capital=10000,
        optimize_metric='sharpe_ratio',
        top_n=5,
        progress_callback=progress,
    )

    print(f"\n  Tested {result['total_combinations']} combinations")

    if result['top_results']:
        print(f"\n{'─' * 60}")
        print("🏆 BEST PARAMETERS")
        print(f"{'─' * 60}")
        for k, v in result['best_params'].items():
            print(f"  {k:20s} = {v}")

        print(f"\n{'─' * 60}")
        print("📊 TRAIN RESULTS (70% of data)")
        print(f"{'─' * 60}")
        tm = result['best_train_metrics']
        print(f"  Trades:        {tm.get('total_trades', 0)}")
        print(f"  Win Rate:      {tm.get('win_rate', 0)}%")
        print(f"  Total Return:  {tm.get('total_return_pct', 0)}%")
        print(f"  Sharpe Ratio:  {tm.get('sharpe_ratio', 0)}")
        print(f"  Max Drawdown:  {tm.get('max_drawdown_pct', 0)}%")
        print(f"  Profit Factor: {tm.get('profit_factor', 0)}")

        print(f"\n{'─' * 60}")
        print("🧪 TEST RESULTS (30% of data — OUT OF SAMPLE)")
        print(f"{'─' * 60}")
        te = result['best_test_metrics']
        print(f"  Trades:        {te.get('total_trades', 0)}")
        print(f"  Win Rate:      {te.get('win_rate', 0)}%")
        print(f"  Total Return:  {te.get('total_return_pct', 0)}%")
        print(f"  Sharpe Ratio:  {te.get('sharpe_ratio', 0)}")
        print(f"  Max Drawdown:  {te.get('max_drawdown_pct', 0)}%")
        print(f"  Profit Factor: {te.get('profit_factor', 0)}")

        print(f"\n{'─' * 60}")
        print("📋 TOP 5 PARAMETER SETS")
        print(f"{'─' * 60}")
        for i, r in enumerate(result['top_results']):
            p = r['params']
            print(f"\n  #{i+1}: score={r['train_score']:.2f} | test={r['test_score']:.2f} | robust={r['robustness']}")
            print(f"       SR={p.get('sr_lookback')}/{p.get('sr_tolerance')} EMA={p.get('ema_fast')}/{p.get('ema_slow')} "
                  f"SL={p.get('sl_atr_mult')}×ATR TP={p.get('tp_atr_mult')}×ATR risk={p.get('risk_pct')}%")
    else:
        print("  ⚠️ No valid parameter sets found")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
