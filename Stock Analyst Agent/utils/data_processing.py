"""CSV ingestion and yfinance market-data helpers."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st
import yfinance as yf

REQUIRED_COLUMNS = ["ticker", "date", "transaction_type", "quantity", "price"]


class TransactionValidationError(ValueError):
    """Raised when the uploaded CSV doesn't match the expected schema."""


def load_transactions(uploaded_file) -> pd.DataFrame:
    """Parse and validate an uploaded transaction-history CSV.

    Expected headers: ticker, date, transaction_type (Buy/Sell), quantity, price.
    Returns a cleaned, chronologically-sorted DataFrame.
    """
    raw_bytes = uploaded_file.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:  # noqa: BLE001
        raise TransactionValidationError(f"Could not read CSV file: {exc}") from exc

    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise TransactionValidationError(
            f"CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected headers: {', '.join(REQUIRED_COLUMNS)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    if df.empty:
        raise TransactionValidationError("The uploaded CSV has no transaction rows.")

    # Normalize types
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["transaction_type"] = df["transaction_type"].astype(str).str.strip().str.title()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    bad_rows = df[
        df["date"].isna()
        | df["quantity"].isna()
        | df["price"].isna()
        | df["ticker"].eq("")
        | ~df["transaction_type"].isin(["Buy", "Sell"])
    ]
    if not bad_rows.empty:
        raise TransactionValidationError(
            f"Found {len(bad_rows)} invalid row(s). Check that date is parseable, "
            "quantity/price are numeric, and transaction_type is 'Buy' or 'Sell'. "
            f"First bad row index: {bad_rows.index[0]}"
        )

    if (df["quantity"] <= 0).any() or (df["price"] < 0).any():
        raise TransactionValidationError(
            "Quantity must be positive and price cannot be negative in every row."
        )

    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_current_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Fetch the latest available close price for each ticker via yfinance."""
    prices: dict[str, float] = {}
    if not tickers:
        return prices
    try:
        data = yf.Tickers(" ".join(tickers))
        for t in tickers:
            price = None
            try:
                fast_info = data.tickers[t].fast_info
                price = fast_info.get("last_price") or fast_info.get("lastPrice")
            except Exception:  # noqa: BLE001
                price = None
            if price is None:
                try:
                    hist = data.tickers[t].history(period="5d")
                    if not hist.empty:
                        price = float(hist["Close"].dropna().iloc[-1])
                except Exception:  # noqa: BLE001
                    price = None
            prices[t] = float(price) if price is not None else float("nan")
    except Exception:  # noqa: BLE001
        for t in tickers:
            prices.setdefault(t, float("nan"))
    return prices


@st.cache_data(ttl=1800, show_spinner=False)
def get_price_history(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Fetch daily close prices for a set of tickers over a date range.

    Returns a wide DataFrame indexed by date with one column per ticker.
    """
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            try:
                closes[t] = raw[t]["Close"]
            except Exception:  # noqa: BLE001
                continue
        out = pd.DataFrame(closes)
    else:
        # Single ticker: columns are simple
        out = raw[["Close"]].rename(columns={"Close": tickers[0]})

    out = out.sort_index().ffill()
    return out
