"""Performance statistics for a monthly return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12


def cagr(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    total = float((1.0 + returns).prod())
    years = len(returns) / PERIODS_PER_YEAR
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(PERIODS_PER_YEAR))


def sharpe(returns: pd.Series, rf_annual: float = 0.06) -> float:
    """Sharpe against an Indian cash rate.

    The default 6% matters: a US-style 0% assumption would overstate Sharpe
    for every Indian strategy, since risk-free rupee deposits have paid
    roughly this over the sample.
    """
    sd = returns.std()
    # Not `sd == 0`: a constant series leaves float residue on the order of
    # 1e-18, which sails past an equality check and yields a Sharpe of ~1e16.
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    rf_monthly = (1.0 + rf_annual) ** (1 / PERIODS_PER_YEAR) - 1.0
    excess = returns - rf_monthly
    return float(excess.mean() / sd * np.sqrt(PERIODS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def hit_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean())


def summary(returns: pd.Series, label: str = "strategy") -> dict:
    return {
        "label": label,
        "periods": len(returns),
        "cagr": cagr(returns),
        "vol": volatility(returns),
        "sharpe": sharpe(returns),
        "max_dd": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "best": float(returns.max()) if len(returns) else float("nan"),
        "worst": float(returns.min()) if len(returns) else float("nan"),
    }


def compare(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("label")
    for col in ("cagr", "vol", "max_dd", "hit_rate"):
        df[col] = (df[col] * 100).round(2)
    df["sharpe"] = df["sharpe"].round(2)
    for col in ("best", "worst"):
        df[col] = (df[col] * 100).round(2)
    return df.rename(
        columns={
            "cagr": "CAGR%",
            "vol": "Vol%",
            "max_dd": "MaxDD%",
            "hit_rate": "Hit%",
            "best": "Best%",
            "worst": "Worst%",
            "sharpe": "Sharpe",
        }
    )


def align_monthly(daily_returns: pd.Series, exit_dates: pd.DatetimeIndex) -> pd.Series:
    """Compound a daily return series onto the backtest's rebalance grid.

    Used to put a benchmark on exactly the same periods as the strategy, so
    the comparison is not quietly measuring different windows.
    """
    out = {}
    edges = list(exit_dates)
    for prev, cur in zip(edges[:-1], edges[1:]):
        window = daily_returns[(daily_returns.index > prev) & (daily_returns.index <= cur)]
        if len(window):
            out[cur] = float((1.0 + window).prod() - 1.0)
    return pd.Series(out)
