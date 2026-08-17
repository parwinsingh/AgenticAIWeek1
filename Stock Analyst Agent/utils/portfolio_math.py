"""Portfolio math: FIFO cost-basis, lifetime metrics, and XIRR."""

from __future__ import annotations

from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from utils.data_processing import get_current_prices


def compute_fifo_positions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute current holdings per ticker using FIFO lot matching.

    Returns columns: ticker, quantity, avg_cost_basis, total_cost,
    realized_pl (from matched sells).
    """
    rows = []
    for ticker, group in transactions.groupby("ticker"):
        lots: deque[list[float]] = deque()  # each lot = [quantity, price]
        realized_pl = 0.0
        for _, txn in group.sort_values("date").iterrows():
            qty = float(txn["quantity"])
            price = float(txn["price"])
            if txn["transaction_type"] == "Buy":
                lots.append([qty, price])
            else:  # Sell
                remaining = qty
                while remaining > 1e-9 and lots:
                    lot_qty, lot_price = lots[0]
                    matched = min(lot_qty, remaining)
                    realized_pl += matched * (price - lot_price)
                    lot_qty -= matched
                    remaining -= matched
                    if lot_qty <= 1e-9:
                        lots.popleft()
                    else:
                        lots[0][0] = lot_qty
                # Selling more than ever bought is ignored beyond available lots
                # (remaining sell quantity simply has no cost basis to reduce).

        total_qty = sum(l[0] for l in lots)
        total_cost = sum(l[0] * l[1] for l in lots)
        avg_cost = (total_cost / total_qty) if total_qty > 1e-9 else 0.0

        rows.append(
            {
                "ticker": ticker,
                "quantity": total_qty,
                "avg_cost_basis": avg_cost,
                "total_cost": total_cost,
                "realized_pl": realized_pl,
            }
        )

    result = pd.DataFrame(rows, columns=["ticker", "quantity", "avg_cost_basis", "total_cost", "realized_pl"])
    # Keep only tickers with an open position for the current holdings view
    return result[result["quantity"] > 1e-9].reset_index(drop=True)


def attach_market_data(positions: pd.DataFrame, prices: dict[str, float]) -> pd.DataFrame:
    """Attach current price / market value / unrealized P&L columns to FIFO positions."""
    if positions.empty:
        return positions.assign(current_price=[], market_value=[], unrealized_pl=[], unrealized_pl_pct=[])

    out = positions.copy()
    out["current_price"] = out["ticker"].map(lambda t: prices.get(t, float("nan")))
    out["market_value"] = out["quantity"] * out["current_price"]
    out["unrealized_pl"] = out["market_value"] - out["total_cost"]
    out["unrealized_pl_pct"] = np.where(
        out["total_cost"] > 0, out["unrealized_pl"] / out["total_cost"] * 100, 0.0
    )
    return out


def get_holdings_view(transactions: pd.DataFrame) -> pd.DataFrame:
    """FIFO positions + live prices + unrealized P&L, ready for display."""
    positions = compute_fifo_positions(transactions)
    prices = get_current_prices(tuple(sorted(positions["ticker"].unique())))
    return attach_market_data(positions, prices)


def compute_lifetime_metrics(transactions: pd.DataFrame, current_value: float) -> dict:
    """Lifetime investment / proceeds / return metrics across the whole history."""
    buys = transactions[transactions["transaction_type"] == "Buy"]
    sells = transactions[transactions["transaction_type"] == "Sell"]

    total_investment = float((buys["quantity"] * buys["price"]).sum())
    total_sells = float((sells["quantity"] * sells["price"]).sum())

    total_return = (current_value + total_sells) - total_investment
    total_return_pct = (total_return / total_investment * 100) if total_investment > 0 else 0.0

    return {
        "total_investment": total_investment,
        "total_sells": total_sells,
        "current_value": current_value,
        "total_return": total_return,
        "total_return_pct": total_return_pct,
    }


def compute_xirr(transactions: pd.DataFrame, current_value: float, as_of: datetime | None = None) -> float | None:
    """Compute the XIRR (annualized IRR on irregular cash flows) for the portfolio.

    Buys are outflows (negative), sells are inflows (positive), and the current
    portfolio value is treated as a final inflow as of today.
    """
    as_of = as_of or datetime.now()

    flows = []
    for _, txn in transactions.iterrows():
        amount = txn["quantity"] * txn["price"]
        amount = -amount if txn["transaction_type"] == "Buy" else amount
        flows.append((pd.Timestamp(txn["date"]).to_pydatetime(), float(amount)))

    if current_value > 0:
        flows.append((as_of, float(current_value)))

    if len(flows) < 2:
        return None

    flows.sort(key=lambda x: x[0])
    t0 = flows[0][0]
    days = np.array([(d - t0).days for d, _ in flows], dtype=float)
    amounts = np.array([a for _, a in flows], dtype=float)

    # Need at least one sign change to have a solvable IRR
    if not (np.any(amounts > 0) and np.any(amounts < 0)):
        return None

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** (days / 365.0)))

    try:
        # Search a wide but bounded range for a root; expand if needed
        low, high = -0.9999, 10.0
        f_low, f_high = npv(low), npv(high)
        if f_low * f_high > 0:
            high = 100.0
            f_high = npv(high)
            if f_low * f_high > 0:
                return None
        rate = brentq(npv, low, high, maxiter=500)
        return float(rate)
    except Exception:  # noqa: BLE001
        return None


def compute_holdings_over_time(transactions: pd.DataFrame, price_history: pd.DataFrame) -> pd.Series:
    """Compute total portfolio market value for each date in price_history's index.

    Holdings quantity per ticker as of a date = cumulative buys - cumulative
    sells up to (and including) that date.
    """
    if price_history.empty:
        return pd.Series(dtype=float)

    dates = price_history.index
    tickers = [t for t in price_history.columns if t in transactions["ticker"].unique()]

    total_value = pd.Series(0.0, index=dates)
    for ticker in tickers:
        t_txns = transactions[transactions["ticker"] == ticker].sort_values("date")
        signed_qty = t_txns["quantity"].where(t_txns["transaction_type"] == "Buy", -t_txns["quantity"])
        qty_by_date = signed_qty.groupby(t_txns["date"]).sum().cumsum()
        # Reindex to the price-history calendar, carrying the last known quantity forward
        qty_on_dates = qty_by_date.reindex(dates.union(qty_by_date.index)).ffill().fillna(0.0)
        qty_on_dates = qty_on_dates.reindex(dates).ffill().fillna(0.0)
        total_value = total_value.add(qty_on_dates * price_history[ticker].reindex(dates).ffill(), fill_value=0.0)

    return total_value
