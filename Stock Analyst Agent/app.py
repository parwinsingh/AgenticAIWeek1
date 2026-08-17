"""Stock Analyst Agent — main Streamlit entry point."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from components import tab_chat, tab_performance, tab_portfolio, tab_upload

load_dotenv()

st.set_page_config(
    page_title="Stock Analyst Agent",
    page_icon="📈",
    layout="wide",
)

st.session_state.setdefault("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))

with st.sidebar:
    st.title("📈 Stock Analyst Agent")
    st.caption("Personal US-equity portfolio tracker, powered by yfinance + OpenAI.")

    api_key_input = st.text_input(
        "OpenAI API key",
        value=st.session_state["openai_api_key"],
        type="password",
        help="Get a key at platform.openai.com. Falls back to OPENAI_API_KEY in .env.",
    )
    st.session_state["openai_api_key"] = api_key_input

    st.divider()
    if "transactions" in st.session_state:
        txns = st.session_state["transactions"]
        st.metric("Transactions loaded", len(txns))
        st.caption(f"{txns['ticker'].nunique()} unique ticker(s)")
    else:
        st.caption("No transactions loaded yet — start in Tab 1.")

st.title("Personal Stock Analyst Agent")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📤 Data Upload", "🧮 Consolidated Portfolio", "📊 Historical Performance", "💬 AI Analyst"]
)

with tab1:
    tab_upload.render()

with tab2:
    tab_portfolio.render()

with tab3:
    tab_performance.render()

with tab4:
    tab_chat.render()
