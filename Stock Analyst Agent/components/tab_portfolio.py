"""Tab 2: Consolidated Portfolio View."""

from __future__ import annotations

import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.llm_agent import LLMNotConfiguredError, generate_portfolio_summary
from utils.portfolio_math import get_holdings_view

# Fixed categorical order (validated for CVD-safe adjacent pairs), capped at 7
# slices + an "Other" bucket so no chart exceeds the validated series count.
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#898781",  # muted gray for "Other"
]


def _build_pie_chart(holdings: pd.DataFrame) -> go.Figure:
    df = holdings.sort_values("market_value", ascending=False).reset_index(drop=True)
    max_slices = 7
    if len(df) > max_slices:
        head = df.iloc[:max_slices]
        other_value = df.iloc[max_slices:]["market_value"].sum()
        labels = list(head["ticker"]) + ["Other"]
        values = list(head["market_value"]) + [other_value]
    else:
        labels = list(df["ticker"])
        values = list(df["market_value"])

    colors = CATEGORICAL_COLORS[: len(labels) - 1] + [CATEGORICAL_COLORS[-1]] if len(labels) > max_slices else CATEGORICAL_COLORS[: len(labels)]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=colors, line=dict(color="#fcfcfb", width=2)),
                textinfo="label+percent",
                textposition="outside",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        margin=dict(t=20, b=20, l=20, r=20),
        height=420,
        annotations=[
            dict(
                text=f"${values and sum(values):,.0f}",
                x=0.5,
                y=0.5,
                font_size=18,
                showarrow=False,
            )
        ],
    )
    return fig


def render() -> None:
    st.subheader("Consolidated Portfolio View")

    if "transactions" not in st.session_state:
        st.info("Upload transaction history in **Tab 1** first.")
        return

    transactions = st.session_state["transactions"]
    holdings = get_holdings_view(transactions)

    if holdings.empty:
        st.warning("No open positions — all shares may have been sold.")
        return

    total_value = float(holdings["market_value"].sum())

    col_chart, col_metric = st.columns([2, 1])
    with col_chart:
        st.plotly_chart(_build_pie_chart(holdings), width="stretch")
    with col_metric:
        st.metric("Total Current Portfolio Value", f"${total_value:,.2f}")
        total_unrealized = float(holdings["unrealized_pl"].sum())
        st.metric(
            "Total Unrealized P/L",
            f"${total_unrealized:,.2f}",
            delta=f"{(total_unrealized / holdings['total_cost'].sum() * 100) if holdings['total_cost'].sum() else 0:+.2f}%",
        )

    st.markdown("#### Stock-wise breakdown")
    display_df = holdings.rename(
        columns={
            "ticker": "Ticker",
            "quantity": "Current Quantity",
            "avg_cost_basis": "Avg Cost Basis (FIFO)",
            "current_price": "Current Price",
            "unrealized_pl": "Unrealized P/L ($)",
            "unrealized_pl_pct": "Unrealized P/L (%)",
        }
    )[
        [
            "Ticker",
            "Current Quantity",
            "Avg Cost Basis (FIFO)",
            "Current Price",
            "Unrealized P/L ($)",
            "Unrealized P/L (%)",
        ]
    ]
    st.dataframe(
        display_df.style.format(
            {
                "Current Quantity": "{:,.2f}",
                "Avg Cost Basis (FIFO)": "${:,.2f}",
                "Current Price": "${:,.2f}",
                "Unrealized P/L ($)": "${:,.2f}",
                "Unrealized P/L (%)": "{:+.2f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### 🤖 AI Portfolio Insight")
    holdings_hash = hashlib.md5(pd.util.hash_pandas_object(holdings).values.tobytes()).hexdigest()
    cache = st.session_state.get("ai_summary_cache")

    if cache and cache.get("hash") == holdings_hash:
        st.info(cache["text"])
    else:
        if st.button("Generate AI insight", key="gen_portfolio_summary"):
            try:
                with st.spinner("Asking OpenAI for a portfolio read..."):
                    summary = generate_portfolio_summary(holdings, total_value)
                st.session_state["ai_summary_cache"] = {"hash": holdings_hash, "text": summary}
                st.rerun()
            except LLMNotConfiguredError as exc:
                st.warning(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"OpenAI request failed: {exc}")
        else:
            st.caption("Click to get an OpenAI-generated health & concentration-risk summary.")
