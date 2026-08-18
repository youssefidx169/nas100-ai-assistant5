"""
NAS100 AI Trading Assistant — Premium Dashboard
A professional backtesting platform for the NAS100 S/R breakout strategy.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.data_provider import download_nas100, get_data_summary, get_available_intervals
from utils.support_resistance import detect_support_resistance
from utils.trade_signals import generate_trade_signals
from utils.backtest import backtest_strategy, format_metrics_display
from utils.optimizer import optimize_strategy, get_default_params, get_default_param_grid
from utils.plots import (
    plot_candlestick_with_trades,
    plot_equity_curve,
    plot_trade_distribution,
    plot_monthly_returns,
    plot_r_multiples,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAS100 AI Trading Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark premium theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #0a0a1a 100%);
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 30, 60, 0.8), rgba(20, 20, 50, 0.9));
        border: 1px solid rgba(100, 100, 255, 0.15);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(100, 100, 255, 0.15);
    }
    .kpi-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00e5ff;
        line-height: 1.2;
    }
    .kpi-value.positive { color: #00e676; }
    .kpi-value.negative { color: #ff1744; }
    .kpi-value.neutral { color: #ffc107; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.85);
        padding: 8px 0;
        border-bottom: 1px solid rgba(100, 100, 255, 0.15);
        margin-bottom: 12px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d24 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(100, 100, 255, 0.1);
    }

    /* Ensure sidebar toggle button is ALWAYS visible & styled brightly */
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        color: #00e5ff !important;
        background: rgba(20, 20, 50, 0.9) !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        border-radius: 8px !important;
        padding: 6px 12px !important;
        margin: 10px !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.3) !important;
    }

    /* Hide default streamlit branding except toggle header */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(20, 20, 50, 0.5);
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        color: rgba(255, 255, 255, 0.6);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(0, 229, 255, 0.15);
        color: #00e5ff;
    }

    /* Table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


def render_kpi(label: str, value: str, color_class: str = ""):
    """Render a KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
    </div>
    """


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 10px 0 10px 0;">
    <h1 style="margin: 0; font-size: 2rem; background: linear-gradient(135deg, #00e5ff, #7c4dff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; letter-spacing: -0.5px;">
        📊 NAS100 AI Trading Assistant
    </h1>
    <p style="color: rgba(255,255,255,0.4); font-size: 0.85rem; margin-top: 4px;">
        Support/Resistance Breakout Strategy • Multi-Indicator Confirmation • Walk-Forward Optimization
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Configuration Panel (Sidebar & Main Expander) ───────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Strategy & Risk Settings")

    st.markdown("---")
    st.markdown("##### 📡 Data Source")

    data_source = st.radio(
        "Source",
        ["Auto-Download (yfinance)", "Upload CSV"],
        index=0,
        key="sidebar_data_source",
    )

    if data_source == "Auto-Download (yfinance)":
        interval = st.selectbox(
            "Timeframe",
            ["1d", "1h", "5m", "15m", "30m", "1wk"],
            index=0,
            key="sidebar_interval",
        )
        period_days = st.slider("History (days)", 30, 3650, 730, key="sidebar_period_days")
        force_refresh = st.checkbox("Force refresh data", value=False, key="sidebar_force_refresh")
    else:
        uploaded_file = st.file_uploader("Upload NAS100 CSV", type=["csv"], key="sidebar_upload")

    st.markdown("---")
    st.markdown("##### 💰 Capital & Risk Management")

    starting_capital = st.number_input("Starting Capital ($)", 1000, 1000000, 10000, step=1000, key="sidebar_capital")
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, step=0.25, key="sidebar_risk")
    sl_atr_mult = st.slider("Stop Loss (ATR ×)", 0.5, 4.0, 1.5, step=0.25, key="sidebar_sl")
    tp_atr_mult = st.slider("Take Profit (ATR ×)", 1.0, 8.0, 3.0, step=0.25, key="sidebar_tp")
    use_trailing = st.checkbox("Trailing Stop", value=True, key="sidebar_trailing")
    commission = st.number_input("Commission per Trade ($)", 0.0, 20.0, 2.0, step=0.5, key="sidebar_commission")

    st.markdown("---")
    st.markdown("##### 📐 Strategy Parameters")

    sr_lookback = st.slider("S/R Pivot Lookback", 2, 15, 5, key="sidebar_sr_lookback")
    sr_tolerance = st.slider("S/R Cluster Tolerance (%)", 0.1, 2.0, 0.5, step=0.1, key="sidebar_sr_tolerance")

    st.markdown("---")
    st.markdown("##### 📈 Indicators")

    ema_fast = st.number_input("EMA Fast Period", 3, 20, 9, key="sidebar_ema_fast")
    ema_slow = st.number_input("EMA Slow Period", 15, 50, 21, key="sidebar_ema_slow")
    use_volume = st.checkbox("Volume Confirmation", value=True, key="sidebar_use_vol")
    vol_threshold = st.slider("Volume Spike Threshold (×)", 1.0, 3.0, 1.5, step=0.1, key="sidebar_vol_thresh")

# Direct in-page config toggle button/expander so users never get lost if sidebar is closed
with st.expander("⚙️ Click to Open / Quick Edit: Risk, Capital, Stop Loss & Strategy Settings", expanded=False):
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown("**💰 Risk & Capital**")
        st.info(f"Capital: **${starting_capital:,.2f}** | Risk: **{risk_pct}%**")
        st.info(f"SL: **{sl_atr_mult}× ATR** | TP: **{tp_atr_mult}× ATR** | Trailing: **{use_trailing}**")

    with col_c2:
        st.markdown("**📐 Indicators & S/R**")
        st.info(f"EMA Fast: **{ema_fast}** | EMA Slow: **{ema_slow}**")
        st.info(f"S/R Lookback: **{sr_lookback}** | Cluster Tol: **{sr_tolerance}%**")

    with col_c3:
        st.markdown("**📡 Data & Costs**")
        if data_source == "Auto-Download (yfinance)":
            st.info(f"Timeframe: **{interval}** | History: **{period_days} days**")
        else:
            st.info("Source: **Custom CSV Upload**")
        st.info(f"Commission: **${commission:.2f}** / trade")
    
    st.caption("💡 Note: Use the left sidebar menu (click top-left arrow `>` if closed) to customize all settings live!")

# ─── Load Data ───────────────────────────────────────────────────────────────
df = None

if data_source == "Auto-Download (yfinance)":
    with st.spinner("📡 Downloading NAS100 data..."):
        try:
            df = download_nas100(
                interval=interval,
                period_days=period_days,
                force_refresh=force_refresh,
            )
            summary = get_data_summary(df)

            # Data summary bar
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(render_kpi("Data Points", f"{summary['rows']:,}"), unsafe_allow_html=True)
            with col2:
                st.markdown(render_kpi("Date Range", f"{summary['start'][:10]}"), unsafe_allow_html=True)
            with col3:
                st.markdown(render_kpi("To", f"{summary['end'][:10]}"), unsafe_allow_html=True)
            with col4:
                st.markdown(render_kpi("Price Range", summary['price_range']), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Failed to download data: {e}")
else:
    if 'uploaded_file' in dir() and uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'Date' in df.columns:
            df.rename(columns={'Date': 'datetime'}, inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime'])
        st.success(f"✅ Loaded {len(df)} rows from uploaded CSV")
    else:
        st.info("📁 Please upload a CSV file with columns: datetime, Open, High, Low, Close, Volume")


# ─── Run Strategy ────────────────────────────────────────────────────────────
if df is not None and len(df) > 30:

    # Create tabs
    tab_overview, tab_trades, tab_optimize, tab_data = st.tabs([
        "📊 Overview", "📋 Trade Log", "🔄 Optimization", "📁 Raw Data"
    ])

    # ── Detect S/R Zones ──
    zones = detect_support_resistance(df, lookback=sr_lookback, tolerance_pct=sr_tolerance)

    # ── Generate Signals ──
    indicator_config = {
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'vol_threshold': vol_threshold,
    }

    signals = generate_trade_signals(
        df, zones,
        config=indicator_config,
        use_volume=use_volume,
    )

    # ── Run Backtest ──
    if not signals.empty:
        trades, equity, metrics = backtest_strategy(
            df, signals,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
            risk_pct=risk_pct,
            starting_capital=starting_capital,
            commission_per_trade=commission,
            use_trailing_stop=use_trailing,
        )
    else:
        trades = pd.DataFrame()
        equity = [starting_capital]
        metrics = {
            'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0,
            'sharpe_ratio': 0, 'max_drawdown_pct': 0, 'profit_factor': 0,
            'total_pnl': 0, 'expectancy': 0, 'final_equity': starting_capital,
            'avg_win': 0, 'avg_loss': 0, 'largest_win': 0, 'largest_loss': 0,
            'avg_r_multiple': 0, 'max_consecutive_wins': 0,
            'max_consecutive_losses': 0, 'winning_trades': 0,
            'losing_trades': 0, 'max_drawdown_value': 0,
            'sortino_ratio': 0, 'calmar_ratio': 0,
            'avg_trade_duration': 'N/A',
        }

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)

        # KPI Row 1
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            pnl = metrics['total_pnl']
            cls = 'positive' if pnl > 0 else 'negative' if pnl < 0 else 'neutral'
            st.markdown(render_kpi("Total P&L", f"${pnl:,.2f}", cls), unsafe_allow_html=True)
        with k2:
            ret = metrics['total_return_pct']
            cls = 'positive' if ret > 0 else 'negative' if ret < 0 else 'neutral'
            st.markdown(render_kpi("Total Return", f"{ret}%", cls), unsafe_allow_html=True)
        with k3:
            wr = metrics['win_rate']
            cls = 'positive' if wr >= 50 else 'negative'
            st.markdown(render_kpi("Win Rate", f"{wr}%", cls), unsafe_allow_html=True)
        with k4:
            st.markdown(render_kpi("Total Trades", str(metrics['total_trades'])), unsafe_allow_html=True)
        with k5:
            sr = metrics['sharpe_ratio']
            cls = 'positive' if sr > 1 else 'neutral' if sr > 0 else 'negative'
            st.markdown(render_kpi("Sharpe Ratio", f"{sr:.2f}", cls), unsafe_allow_html=True)
        with k6:
            mdd = metrics['max_drawdown_pct']
            cls = 'negative' if mdd < -10 else 'neutral' if mdd < 0 else 'positive'
            st.markdown(render_kpi("Max Drawdown", f"{mdd}%", cls), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # KPI Row 2
        k7, k8, k9, k10, k11, k12 = st.columns(6)
        with k7:
            pf = metrics['profit_factor']
            cls = 'positive' if pf > 1.5 else 'neutral' if pf > 1 else 'negative'
            pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
            st.markdown(render_kpi("Profit Factor", pf_str, cls), unsafe_allow_html=True)
        with k8:
            exp = metrics['expectancy']
            cls = 'positive' if exp > 0 else 'negative'
            st.markdown(render_kpi("Expectancy", f"${exp:,.2f}", cls), unsafe_allow_html=True)
        with k9:
            st.markdown(render_kpi("Avg R-Multiple", f"{metrics['avg_r_multiple']:.2f}"), unsafe_allow_html=True)
        with k10:
            st.markdown(render_kpi("Final Equity", f"${metrics['final_equity']:,.2f}"), unsafe_allow_html=True)
        with k11:
            st.markdown(render_kpi("Consec. Wins", str(metrics['max_consecutive_wins'])), unsafe_allow_html=True)
        with k12:
            st.markdown(render_kpi("Consec. Losses", str(metrics['max_consecutive_losses'])), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Candlestick Chart
        st.markdown('<div class="section-header">Price Chart with Trades</div>', unsafe_allow_html=True)
        fig_candle = plot_candlestick_with_trades(df, trades, zones)
        st.plotly_chart(fig_candle, use_container_width=True)

        # Equity Curve
        st.markdown('<div class="section-header">Equity Curve & Drawdown</div>', unsafe_allow_html=True)
        fig_equity = plot_equity_curve(equity, starting_capital)
        st.plotly_chart(fig_equity, use_container_width=True)

        # Bottom row: distribution + R-multiples
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown('<div class="section-header">PnL Distribution</div>', unsafe_allow_html=True)
            fig_dist = plot_trade_distribution(trades)
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_right:
            st.markdown('<div class="section-header">R-Multiples per Trade</div>', unsafe_allow_html=True)
            fig_r = plot_r_multiples(trades)
            st.plotly_chart(fig_r, use_container_width=True)

        # Monthly returns
        if not trades.empty:
            st.markdown('<div class="section-header">Monthly Returns</div>', unsafe_allow_html=True)
            fig_monthly = plot_monthly_returns(trades)
            st.plotly_chart(fig_monthly, use_container_width=True)

        # S/R Zones table
        if zones:
            st.markdown('<div class="section-header">Detected Support & Resistance Zones</div>', unsafe_allow_html=True)
            zones_df = pd.DataFrame(zones)
            zones_df['level'] = zones_df['level'].round(2)
            zones_df['support'] = zones_df['support'].round(2)
            zones_df['resistance'] = zones_df['resistance'].round(2)
            zones_df['strength'] = (zones_df['strength'] * 100).round(1).astype(str) + '%'
            st.dataframe(zones_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: TRADE LOG
    # ═══════════════════════════════════════════════════════════════════════
    with tab_trades:
        st.markdown('<div class="section-header">Trade Log</div>', unsafe_allow_html=True)

        if not trades.empty:
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Winning Trades", metrics['winning_trades'])
            with col2:
                st.metric("Losing Trades", metrics['losing_trades'])
            with col3:
                st.metric("Avg Duration", metrics['avg_trade_duration'])

            # Display trades
            display_trades = trades.copy()
            for col in ['entry_price', 'exit_price', 'sl_price', 'tp_price', 'gross_pnl', 'net_pnl']:
                if col in display_trades.columns:
                    display_trades[col] = display_trades[col].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_trades, use_container_width=True, hide_index=True)

            # Native Streamlit Download Button
            csv_data = trades.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Trades CSV",
                data=csv_data,
                file_name="nas100_trades.csv",
                mime="text/csv",
                type="primary",
            )
        else:
            st.warning("No trades were generated. Try adjusting parameters.")

        # Signals table
        if not signals.empty:
            st.markdown('<div class="section-header">Generated Signals</div>', unsafe_allow_html=True)
            st.dataframe(signals, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════
    with tab_optimize:
        st.markdown('<div class="section-header">Walk-Forward Optimization</div>', unsafe_allow_html=True)

        st.markdown("""
        <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">
            Splits data into 70% train / 30% test. Searches over parameter grid to find the 
            best risk-adjusted parameters. Results are ranked by Sharpe ratio × robustness 
            (test/train consistency) to avoid overfitting.
        </p>
        """, unsafe_allow_html=True)

        optimize_metric = st.selectbox(
            "Optimize for",
            ["sharpe_ratio", "total_return_pct", "profit_factor", "expectancy"],
            index=0,
            key="opt_metric_select",
        )

        if st.button("🚀 Run Optimization", type="primary", key="btn_run_opt"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"Testing combination {current}/{total}...")

            with st.spinner("Optimizing strategy... Please wait."):
                st.session_state['opt_result'] = optimize_strategy(
                    df,
                    train_ratio=0.7,
                    starting_capital=starting_capital,
                    optimize_metric=optimize_metric,
                    top_n=5,
                    progress_callback=update_progress,
                )

            progress_bar.empty()
            status_text.empty()

        # Display results if available in session state
        opt_result = st.session_state.get('opt_result')
        if opt_result:
            if opt_result.get('top_results'):
                st.success(f"✅ Tested {opt_result['total_combinations']} parameter combinations")

                # Best parameters
                st.markdown('<div class="section-header">🏆 Best Parameters</div>', unsafe_allow_html=True)
                best = opt_result['best_params']
                param_cols = st.columns(len(best))
                for i, (key, val) in enumerate(best.items()):
                    with param_cols[i % len(param_cols)]:
                        st.metric(key.replace('_', ' ').title(), val)

                # Train vs Test comparison
                st.markdown('<div class="section-header">Train vs Test Results</div>', unsafe_allow_html=True)
                col_train, col_test = st.columns(2)

                train_m = opt_result['best_train_metrics']
                test_m = opt_result['best_test_metrics']

                with col_train:
                    st.markdown("**📊 Training Set (70%)**")
                    st.metric("Sharpe Ratio", train_m.get('sharpe_ratio', 'N/A'))
                    st.metric("Win Rate", f"{train_m.get('win_rate', 0)}%")
                    st.metric("Total Return", f"{train_m.get('total_return_pct', 0)}%")
                    st.metric("Max Drawdown", f"{train_m.get('max_drawdown_pct', 0)}%")
                    st.metric("Profit Factor", train_m.get('profit_factor', 0))

                with col_test:
                    st.markdown("**🧪 Test Set (30%) — Out of Sample**")
                    st.metric("Sharpe Ratio", test_m.get('sharpe_ratio', 'N/A'))
                    st.metric("Win Rate", f"{test_m.get('win_rate', 0)}%")
                    st.metric("Total Return", f"{test_m.get('total_return_pct', 0)}%")
                    st.metric("Max Drawdown", f"{test_m.get('max_drawdown_pct', 0)}%")
                    st.metric("Profit Factor", test_m.get('profit_factor', 0))

                # Top 5 results
                st.markdown('<div class="section-header">Top 5 Parameter Sets</div>', unsafe_allow_html=True)
                top_data = []
                for r in opt_result['top_results']:
                    row = {**r['params']}
                    row['train_score'] = round(r['train_score'], 2) if isinstance(r['train_score'], float) else r['train_score']
                    row['test_score'] = round(r['test_score'], 2) if isinstance(r['test_score'], float) else r['test_score']
                    row['robustness'] = r['robustness']
                    row['train_trades'] = r['train_trades']
                    row['test_trades'] = r['test_trades']
                    top_data.append(row)
                st.dataframe(pd.DataFrame(top_data), use_container_width=True, hide_index=True)

            else:
                st.warning("⚠️ No valid parameter combinations found. Try different data or wider parameter ranges.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: RAW DATA
    # ═══════════════════════════════════════════════════════════════════════
    with tab_data:
        st.markdown('<div class="section-header">Raw OHLCV Data</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

        raw_csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Raw NAS100 Data CSV",
            data=raw_csv_data,
            file_name="nas100_raw_data.csv",
            mime="text/csv",
            type="secondary",
        )

elif df is not None:
    st.warning("⚠️ Not enough data points. Need at least 30 bars for meaningful analysis.")
