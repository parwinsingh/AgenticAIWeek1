"""OpenAI LLM integration: portfolio health summary + conversational analyst."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from openai import OpenAI

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class LLMNotConfiguredError(RuntimeError):
    """Raised when no OpenAI API key is available."""


def get_api_key() -> str | None:
    return st.session_state.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")


def get_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        raise LLMNotConfiguredError(
            "No OpenAI API key found. Add it in the sidebar or set OPENAI_API_KEY in your .env file."
        )
    return OpenAI(api_key=api_key)


def _format_holdings_for_prompt(holdings: pd.DataFrame, total_value: float) -> str:
    if holdings.empty:
        return "No open positions."
    lines = []
    for _, row in holdings.iterrows():
        weight = (row["market_value"] / total_value * 100) if total_value > 0 else 0.0
        lines.append(
            f"- {row['ticker']}: {row['quantity']:.2f} shares, "
            f"market value ${row['market_value']:,.2f} ({weight:.1f}% of portfolio), "
            f"unrealized {row['unrealized_pl_pct']:+.1f}%"
        )
    return "\n".join(lines)


def generate_portfolio_summary(holdings: pd.DataFrame, total_value: float) -> str:
    """Ask OpenAI for a 2-3 sentence portfolio health / concentration-risk summary."""
    client = get_client()
    holdings_text = _format_holdings_for_prompt(holdings, total_value)

    prompt = (
        "You are a concise portfolio analyst. Given the current holdings below, "
        "write a 2-3 sentence summary covering overall portfolio health and any "
        "concentration risk (e.g. single-stock or sector overweight). Be specific "
        "with tickers and percentages where useful. Do not give financial advice "
        "disclaimers, just the analysis.\n\n"
        f"Total portfolio value: ${total_value:,.2f}\n"
        f"Holdings:\n{holdings_text}"
    )

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=220,
    )
    return response.choices[0].message.content.strip()


def build_portfolio_context(
    transactions: pd.DataFrame,
    holdings: pd.DataFrame,
    lifetime_metrics: dict,
    xirr: float | None,
) -> str:
    """Build a compact text context describing the portfolio for the chat agent."""
    txn_summary = transactions.copy()
    txn_summary["date"] = pd.to_datetime(txn_summary["date"]).dt.strftime("%Y-%m-%d")
    txn_lines = "\n".join(
        f"{r.date} | {r.ticker} | {r.transaction_type} | qty {r.quantity} | price ${r.price:.2f}"
        for r in txn_summary.itertuples()
    )

    holdings_lines = "\n".join(
        f"{r.ticker}: {r.quantity:.2f} shares @ avg cost ${r.avg_cost_basis:.2f}, "
        f"current price ${r.current_price:.2f}, market value ${r.market_value:,.2f}, "
        f"unrealized {r.unrealized_pl_pct:+.1f}%"
        for r in holdings.itertuples()
    ) if not holdings.empty else "No open positions."

    xirr_text = f"{xirr * 100:.2f}%" if xirr is not None else "N/A (insufficient cash-flow history)"

    return (
        "=== TRANSACTION HISTORY ===\n"
        f"{txn_lines}\n\n"
        "=== CURRENT HOLDINGS ===\n"
        f"{holdings_lines}\n\n"
        "=== PERFORMANCE METRICS ===\n"
        f"Total Investment (lifetime): ${lifetime_metrics['total_investment']:,.2f}\n"
        f"Total Sell Proceeds (lifetime): ${lifetime_metrics['total_sells']:,.2f}\n"
        f"Current Portfolio Value: ${lifetime_metrics['current_value']:,.2f}\n"
        f"Total Return: ${lifetime_metrics['total_return']:,.2f} ({lifetime_metrics['total_return_pct']:+.2f}%)\n"
        f"XIRR: {xirr_text}\n"
    )


SYSTEM_PROMPT_TEMPLATE = (
    "You are an AI Stock Analyst Agent helping a retail investor understand their "
    "own US equity portfolio. You have been given their full transaction history, "
    "current holdings, and performance metrics below. Answer questions grounded "
    "strictly in this data. If asked about intraday moves ('why is my portfolio "
    "down today'), reason from the holdings/current prices you were given and be "
    "explicit that you don't have real-time news access, but you may speculate "
    "based on general market knowledge. Be concise, use specific numbers and "
    "tickers, and never fabricate data not present in the context.\n\n"
    "{context}"
)


def stream_chat_response(messages: list[dict], portfolio_context: str):
    """Yield streaming text chunks from OpenAI for the chat tab."""
    client = get_client()
    system_message = {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=portfolio_context)}

    stream = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[system_message, *messages],
        temperature=0.5,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
