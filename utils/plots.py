"""
Interactive Plotly charts for the NAS100 trading dashboard.
Includes candlestick charts, equity curves, drawdown visualization,
monthly returns heatmap, and trade distribution.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_candlestick_with_trades(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    zones: list = None,
    title: str = "NAS100 Price Chart with Trades",
) -> go.Figure:
    """
    Interactive candlestick chart with S/R zones and trade markers.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(title, 'Volume'),
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1,
    )

    # Volume bars
    colors = ['#26a69a' if c >= o else '#ef5350'
              for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(
            x=df['datetime'],
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5,
        ),
        row=2, col=1,
    )

    # S/R Zones
    if zones:
        for zone in zones[:10]:  # Top 10 zones
            opacity = 0.1 + (zone.get('strength', 0.5) * 0.2)
            color = 'rgba(255, 152, 0, {})'.format(opacity)
            if zone['type'] == 'support':
                color = 'rgba(76, 175, 80, {})'.format(opacity)

            fig.add_hrect(
                y0=zone['support'], y1=zone['resistance'],
                fillcolor=color,
                line_width=0,
                row=1, col=1,
                annotation_text=f"{zone['type'].title()} ({zone['touches']})",
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color="rgba(255,255,255,0.7)",
            )

    # Trade markers
    if not trades.empty:
        # Entry markers
        longs = trades[trades['direction'] == 'breakout_long']
        shorts = trades[trades['direction'] == 'breakout_short']

        if not longs.empty:
            fig.add_trace(
                go.Scatter(
                    x=longs['entry_time'],
                    y=longs['entry_price'],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='#00e676',
                        line=dict(width=1, color='white'),
                    ),
                    text=[f"Long @ {p:.0f}" for p in longs['entry_price']],
                ),
                row=1, col=1,
            )

        if not shorts.empty:
            fig.add_trace(
                go.Scatter(
                    x=shorts['entry_time'],
                    y=shorts['entry_price'],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='#ff1744',
                        line=dict(width=1, color='white'),
                    ),
                    text=[f"Short @ {p:.0f}" for p in shorts['entry_price']],
                ),
                row=1, col=1,
            )

        # Exit markers
        wins = trades[trades['Result'] == 'Win']
        losses = trades[trades['Result'] == 'Loss']

        if not wins.empty:
            fig.add_trace(
                go.Scatter(
                    x=wins['exit_time'],
                    y=wins['exit_price'],
                    mode='markers',
                    name='Win Exit',
                    marker=dict(
                        symbol='star',
                        size=10,
                        color='#00e676',
                        line=dict(width=1, color='white'),
                    ),
                ),
                row=1, col=1,
            )

        if not losses.empty:
            fig.add_trace(
                go.Scatter(
                    x=losses['exit_time'],
                    y=losses['exit_price'],
                    mode='markers',
                    name='Loss Exit',
                    marker=dict(
                        symbol='x',
                        size=10,
                        color='#ff1744',
                        line=dict(width=1, color='white'),
                    ),
                ),
                row=1, col=1,
            )

    fig.update_layout(
        template='plotly_dark',
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=50, r=50, t=60, b=30),
    )

    return fig


def plot_equity_curve(equity: list, starting_capital: float = 10000.0) -> go.Figure:
    """
    Equity curve with drawdown shading.
    """
    eq = pd.Series(equity)
    rolling_max = eq.cummax()
    drawdown = eq - rolling_max
    drawdown_pct = (drawdown / rolling_max) * 100

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Equity Curve', 'Drawdown (%)'),
        row_heights=[0.65, 0.35],
    )

    # Equity line
    fig.add_trace(
        go.Scatter(
            x=list(range(len(eq))),
            y=eq,
            mode='lines',
            name='Equity',
            line=dict(color='#00e5ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 229, 255, 0.1)',
        ),
        row=1, col=1,
    )

    # Starting capital reference
    fig.add_hline(
        y=starting_capital,
        line_dash="dot",
        line_color="rgba(255,255,255,0.3)",
        annotation_text=f"Start: ${starting_capital:,.0f}",
        annotation_font_size=10,
        row=1, col=1,
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=list(range(len(drawdown_pct))),
            y=drawdown_pct,
            mode='lines',
            name='Drawdown %',
            line=dict(color='#ff1744', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 23, 68, 0.2)',
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template='plotly_dark',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=50, r=50, t=60, b=30),
    )

    fig.update_xaxes(title_text="Trade #", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def plot_trade_distribution(trades: pd.DataFrame) -> go.Figure:
    """
    Histogram of trade PnL distribution.
    """
    if trades.empty:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            title="No trades to display",
        )
        return fig

    wins = trades[trades['Result'] == 'Win']['net_pnl']
    losses = trades[trades['Result'] == 'Loss']['net_pnl']

    fig = go.Figure()

    if not wins.empty:
        fig.add_trace(
            go.Histogram(
                x=wins,
                name='Winning Trades',
                marker_color='#00e676',
                opacity=0.7,
                nbinsx=20,
            )
        )

    if not losses.empty:
        fig.add_trace(
            go.Histogram(
                x=losses,
                name='Losing Trades',
                marker_color='#ff1744',
                opacity=0.7,
                nbinsx=20,
            )
        )

    fig.update_layout(
        template='plotly_dark',
        title='Trade PnL Distribution',
        xaxis_title='PnL ($)',
        yaxis_title='Count',
        barmode='overlay',
        height=350,
        margin=dict(l=50, r=50, t=60, b=30),
    )

    return fig


def plot_monthly_returns(trades: pd.DataFrame) -> go.Figure:
    """
    Monthly returns heatmap.
    """
    if trades.empty or 'exit_time' not in trades.columns:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', title="No data for monthly returns")
        return fig

    df = trades.copy()
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['year'] = df['exit_time'].dt.year
    df['month'] = df['exit_time'].dt.month

    monthly = df.groupby(['year', 'month'])['net_pnl'].sum().reset_index()
    pivot = monthly.pivot(index='year', columns='month', values='net_pnl').fillna(0)

    # Ensure all months are present
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = 0
    pivot = pivot[sorted(pivot.columns)]

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=month_names,
            y=[str(y) for y in pivot.index],
            colorscale=[
                [0, '#ff1744'],
                [0.5, '#1a1a2e'],
                [1, '#00e676'],
            ],
            text=[[f"${v:,.0f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=11),
            hoverongaps=False,
            colorbar=dict(title="PnL ($)"),
        )
    )

    fig.update_layout(
        template='plotly_dark',
        title='Monthly Returns Heatmap',
        height=300,
        margin=dict(l=50, r=50, t=60, b=30),
    )

    return fig


def plot_r_multiples(trades: pd.DataFrame) -> go.Figure:
    """
    Bar chart of R-multiples per trade.
    """
    if trades.empty:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', title="No trades")
        return fig

    colors = ['#00e676' if r > 0 else '#ff1744' for r in trades['r_multiple']]

    fig = go.Figure(
        data=go.Bar(
            x=list(range(1, len(trades) + 1)),
            y=trades['r_multiple'],
            marker_color=colors,
            name='R-Multiple',
        )
    )

    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        template='plotly_dark',
        title='R-Multiples per Trade',
        xaxis_title='Trade #',
        yaxis_title='R-Multiple',
        height=300,
        margin=dict(l=50, r=50, t=60, b=30),
    )

    return fig


# Keep backward compatibility
def plot_trades(df, trades):
    """Backward-compatible wrapper."""
    return plot_candlestick_with_trades(df, trades)
