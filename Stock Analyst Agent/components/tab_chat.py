"""Tab 4: AI Analyst (Chat)."""

from __future__ import annotations

import streamlit as st

from utils.llm_agent import LLMNotConfiguredError, build_portfolio_context, stream_chat_response
from utils.portfolio_math import compute_lifetime_metrics, compute_xirr, get_holdings_view

SUGGESTED_PROMPTS = [
    "Summarize my historical trading performance",
    "Why might my portfolio be down today?",
    "What's my biggest concentration risk?",
    "Which of my positions has the best unrealized return?",
]


def render() -> None:
    st.subheader("AI Analyst Chat")

    if "transactions" not in st.session_state:
        st.info("Upload transaction history in **Tab 1** first.")
        return

    transactions = st.session_state["transactions"]
    holdings = get_holdings_view(transactions)
    current_value = float(holdings["market_value"].sum()) if not holdings.empty else 0.0
    metrics = st.session_state.get("lifetime_metrics") or compute_lifetime_metrics(transactions, current_value)
    xirr = st.session_state.get("xirr")
    if xirr is None:
        xirr = compute_xirr(transactions, current_value)

    context = build_portfolio_context(transactions, holdings, metrics, xirr)

    st.caption("Ask questions about your holdings, trades, or performance. Context includes your full history.")

    st.session_state.setdefault("chat_messages", [])

    cols = st.columns(len(SUGGESTED_PROMPTS))
    for col, prompt_text in zip(cols, SUGGESTED_PROMPTS):
        if col.button(prompt_text, width="stretch"):
            st.session_state["_pending_prompt"] = prompt_text

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pop("_pending_prompt", None)
    user_input = st.chat_input("Ask your AI Stock Analyst...") or pending

    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                response_text = st.write_stream(
                    stream_chat_response(st.session_state["chat_messages"], context)
                )
            except LLMNotConfiguredError as exc:
                response_text = str(exc)
                st.warning(response_text)
            except Exception as exc:  # noqa: BLE001
                response_text = f"OpenAI request failed: {exc}"
                st.error(response_text)

        st.session_state["chat_messages"].append({"role": "assistant", "content": response_text})

    if st.session_state["chat_messages"] and st.button("Clear chat"):
        st.session_state["chat_messages"] = []
        st.rerun()
