"""Tab 1: Data Upload."""

from __future__ import annotations

import streamlit as st

from utils.data_processing import TransactionValidationError, load_transactions

SAMPLE_CSV = (
    "ticker,date,transaction_type,quantity,price\n"
    "AAPL,2023-01-15,Buy,10,135.21\n"
    "AAPL,2023-06-01,Buy,5,180.12\n"
    "MSFT,2023-02-10,Buy,8,255.30\n"
    "NVDA,2023-03-05,Buy,15,235.50\n"
    "AAPL,2024-01-20,Sell,4,193.00\n"
    "GOOGL,2023-04-18,Buy,12,103.75\n"
)


def render() -> None:
    st.subheader("Upload Transaction History")
    st.caption(
        "Upload a CSV with columns: **ticker, date, transaction_type (Buy/Sell), "
        "quantity, price**. This replaces the current transaction history."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Transaction CSV", type=["csv"], label_visibility="collapsed")
    with col2:
        st.download_button(
            "Download sample CSV",
            data=SAMPLE_CSV,
            file_name="sample_transactions.csv",
            mime="text/csv",
            width="stretch",
        )

    if uploaded_file is not None:
        try:
            df = load_transactions(uploaded_file)
        except TransactionValidationError as exc:
            st.error(str(exc))
            return

        st.session_state["transactions"] = df
        st.session_state.pop("ai_summary_cache", None)
        st.success(f"Loaded {len(df)} transactions across {df['ticker'].nunique()} ticker(s).")

    if "transactions" in st.session_state:
        st.markdown("#### Current transaction history")
        st.dataframe(st.session_state["transactions"], width="stretch", hide_index=True)
        if st.button("Clear uploaded data"):
            for key in ("transactions", "ai_summary_cache", "chat_messages"):
                st.session_state.pop(key, None)
            st.rerun()
    else:
        st.info("No transaction history uploaded yet. Upload a CSV to get started.")
