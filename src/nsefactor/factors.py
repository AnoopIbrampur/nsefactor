"""Cross-sectional factors, computed causally from the adjusted price panel.

Every factor here is priced off daily bars alone. That is a real constraint
worth stating plainly rather than papering over: the bhavcopy carries no
shares outstanding, no book value, and no earnings, so **this module cannot
compute size, value, or quality**. Market cap needs a share count; ROE and
accruals need financial statements. Both require a fundamentals source we do
not have yet.

What daily bars *do* support is the set below -- momentum, low volatility,
short-term reversal, and illiquidity -- all of which are documented
cross-sectional effects that survive out of sample in most markets. It is a
narrower hypothesis than a full multi-factor model, and the results should be
read as such.

Every function takes the panel as of a formation date and looks strictly
backwards. Nothing may touch a bar dated after ``as_of``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Trading-day conventions. NSE averages ~247 sessions a year.
MONTH = 21
YEAR = 252

# Higher score should always mean "expected to outperform", so factors whose
# raw value predicts negatively are negated when combined.
FACTOR_SIGNS = {
    "mom_12_1": +1,  # winners keep winning over 3-12 months
    "mom_6_1": +1,
    "vol_126": -1,  # low-volatility anomaly: calmer stocks earn more per unit risk
    "reversal_21": -1,  # last month's winners give some back
    "illiq_126": +1,  # Amihud: illiquid names carry a premium
}


def _wide(panel: pd.DataFrame, column: str, as_of: pd.Timestamp, lookback: int) -> pd.DataFrame:
    """Dates x ISIN matrix of ``column``, ending at ``as_of`` inclusive."""
    hist = panel[panel["date"] <= as_of]
    days = pd.DatetimeIndex(sorted(hist["date"].unique()))[-lookback:]
    hist = hist[hist["date"].isin(days)]
    return hist.pivot_table(index="date", columns="isin", values=column, aggfunc="last")


def momentum(panel: pd.DataFrame, as_of: pd.Timestamp, lookback: int, skip: int = MONTH) -> pd.Series:
    """Cumulative return over ``lookback`` days, skipping the most recent ``skip``.

    The skip is not decoration. Momentum and short-term reversal point in
    opposite directions at the one-month horizon, so a 12-month window that
    includes last month blends two effects that partially cancel. Dropping the
    final month is the standard 12-1 construction.
    """
    prices = _wide(panel, "adj_close", as_of, lookback + skip + 1)
    if len(prices) < lookback + skip:
        return pd.Series(dtype=float)

    end = prices.iloc[-(skip + 1)]
    start = prices.iloc[0]
    valid = start.notna() & end.notna() & (start > 0)
    return ((end / start) - 1.0).where(valid)


def volatility(panel: pd.DataFrame, as_of: pd.Timestamp, lookback: int = 126) -> pd.Series:
    """Annualised realised volatility of daily returns."""
    prices = _wide(panel, "adj_close", as_of, lookback + 1)
    if len(prices) < lookback // 2:
        return pd.Series(dtype=float)
    rets = prices.pct_change()
    return rets.std() * np.sqrt(YEAR)


def reversal(panel: pd.DataFrame, as_of: pd.Timestamp, lookback: int = MONTH) -> pd.Series:
    """Return over the trailing month. Negated when combined."""
    prices = _wide(panel, "adj_close", as_of, lookback + 1)
    if len(prices) < 2:
        return pd.Series(dtype=float)
    start, end = prices.iloc[0], prices.iloc[-1]
    valid = start.notna() & end.notna() & (start > 0)
    return ((end / start) - 1.0).where(valid)


def illiquidity(panel: pd.DataFrame, as_of: pd.Timestamp, lookback: int = 126) -> pd.Series:
    """Amihud illiquidity: mean |return| per rupee of turnover.

    Scaled up by 1e6 purely to keep the numbers readable; the cross-sectional
    ranking is unaffected by any positive scaling.
    """
    prices = _wide(panel, "adj_close", as_of, lookback + 1)
    turnover = _wide(panel, "turnover", as_of, lookback + 1)
    if len(prices) < 2:
        return pd.Series(dtype=float)

    rets = prices.pct_change().abs()
    turnover = turnover.reindex_like(rets).replace(0, np.nan)
    return (rets / turnover).mean() * 1e6


def zscore(raw: pd.Series, winsor: float = 0.02) -> pd.Series:
    """Cross-sectional z-score, winsorised at both tails.

    Winsorising matters more than it looks. Illiquidity in particular is
    violently right-skewed, and without trimming a couple of near-untraded
    names would dominate the composite by themselves.
    """
    s = raw.dropna()
    if len(s) < 10:
        return pd.Series(dtype=float)
    lo, hi = s.quantile(winsor), s.quantile(1 - winsor)
    clipped = s.clip(lo, hi)
    sd = clipped.std()
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (clipped - clipped.mean()) / sd


def compute_all(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    isins: pd.Index | None = None,
    use: tuple[str, ...] | None = None,
    require_all: bool = True,
) -> pd.DataFrame:
    """All factors for one formation date, raw values plus z-scores.

    ``isins`` restricts the cross-section to the investable universe. This is
    deliberate: z-scores must be computed *within* the universe the portfolio
    can actually buy, or the standardisation is set by micro-caps that will
    never be held.

    ``use`` selects which factors enter the composite. All raw values and
    z-scores are still returned, so a factor can be measured without being
    traded -- which is how a factor gets dropped on training-period evidence
    and then verified as still-dropped out of sample.
    """
    raw = {
        "mom_12_1": momentum(panel, as_of, YEAR - MONTH),
        "mom_6_1": momentum(panel, as_of, 126 - MONTH),
        "vol_126": volatility(panel, as_of, 126),
        "reversal_21": reversal(panel, as_of, MONTH),
        "illiq_126": illiquidity(panel, as_of, 126),
    }
    df = pd.DataFrame(raw)
    if isins is not None:
        df = df.reindex(isins)

    for name, sign in FACTOR_SIGNS.items():
        df[f"z_{name}"] = sign * zscore(df[name])

    selected = tuple(use) if use is not None else tuple(FACTOR_SIGNS)
    unknown = set(selected) - set(FACTOR_SIGNS)
    if unknown:
        raise KeyError(f"unknown factors: {sorted(unknown)}")

    zcols = [f"z_{n}" for n in selected]
    if require_all:
        # With a small factor set, "at least half" means a stock missing
        # momentum can still be ranked on volatility alone -- which is how a
        # company listed six months ago ends up scored as though it had a
        # momentum history. Demand every selected factor.
        enough = df[zcols].notna().all(axis=1)
    else:
        enough = df[zcols].notna().sum(axis=1) >= len(zcols) / 2
    df["composite"] = df[zcols].mean(axis=1, skipna=True).where(enough)
    df["date"] = as_of
    return df
