"""Tab 3: Historical Performance."""

from __future__ import annotations

from datetime import timedelta

import plotly.graph_objects as go
import streamlit as st

from utils.data_processing import get_price_history
from utils.portfolio_math import (
    compute_holdings_over_time,
    compute_lifetime_metrics,
    compute_xirr,
    get_holdings_view,
)

LINE_COLOR = "#2a78d6"  # categorical slot 1 (blue) — single series


def _build_value_chart(series) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line=dict(color=LINE_COLOR, width=2),
                fill="tozeroy",
                fillcolor="rgba(42, 120, 214, 0.10)",
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#e1e0d9", tickprefix="$"),
        showlegend=False,
    )
    return fig


def render() -> None:
    st.subheader("Historical Performance")

    if "transactions" not in st.session_state:
        st.info("Upload transaction history in **Tab 1** first.")
        return

    transactions = st.session_state["transactions"]
    holdings = get_holdings_view(transactions)
    current_value = float(holdings["market_value"].sum()) if not holdings.empty else 0.0

    metrics = compute_lifetime_metrics(transactions, current_value)
    xirr = compute_xirr(transactions, current_value)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Investment (lifetime)", f"${metrics['total_investment']:,.2f}")
    c2.metric("Total Sells (lifetime proceeds)", f"${metrics['total_sells']:,.2f}")
    c3.metric("Current Portfolio Value", f"${metrics['current_value']:,.2f}")
    c4.metric(
        "Total Return",
        f"${metrics['total_return']:,.2f}",
        delta=f"{metrics['total_return_pct']:+.2f}%",
    )
    c5.metric("XIRR", f"{xirr * 100:,.2f}%" if xirr is not None else "N/A")

    st.session_state["lifetime_metrics"] = metrics
    st.session_state["xirr"] = xirr

    st.markdown("#### Portfolio value over time")
    tickers = tuple(sorted(transactions["ticker"].unique()))
    start = transactions["date"].min().date()
    end = (transactions["date"].max().date() + timedelta(days=1))
    from datetime import date as _date

    end = max(end, _date.today())

    with st.spinner("Fetching historical prices..."):
        history = get_price_history(tickers, str(start), str(end + timedelta(days=1)))

    if history.empty:
        st.caption("Historical price chart unavailable (no data returned by yfinance).")
        return

    value_series = compute_holdings_over_time(transactions, history)
    value_series = value_series[value_series.index.date >= start]
    if value_series.empty:
        st.caption("Not enough historical data to chart portfolio value.")
    else:
        st.plotly_chart(_build_value_chart(value_series), width="stretch")
