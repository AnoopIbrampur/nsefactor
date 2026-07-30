"""Value and quality factors, built strictly from what was public at the time.

These are the two factor families the price-only model could not reach, and the
two most associated with the long-horizon investing this repo is aimed at.

Every factor here needs two inputs that must be aligned carefully:

* **fundamentals**, which are known only from their broadcast date onward;
* **market prices**, which are known daily.

The join is where value factors usually go wrong. Earnings yield is
``trailing earnings / market cap``, and it is tempting to compute market cap
from shares outstanding reported in the *same* filing as the earnings. But that
filing was not public until weeks after its period ended, so the correct
pairing is *the latest earnings visible today* against *today's price* -- never
the price that prevailed when the accounts were drawn up.

Sign convention matches :mod:`nsefactor.factors`: a higher score always means
"expected to outperform", so cheapness and quality score positive while
leverage scores negative.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import fundamentals as F

log = logging.getLogger(__name__)

# Caps the earnings-stability score at 1/0.02 = 50, i.e. a 2% coefficient of
# variation counts as "as steady as we can distinguish".
CV_FLOOR = 0.02

FUND_FACTOR_SIGNS = {
    "earnings_yield": +1,  # cheap on earnings
    "book_to_price": +1,  # cheap on assets
    "roe": +1,  # profitable use of equity
    "gross_margin": +1,  # pricing power / business quality
    "debt_to_equity": -1,  # leverage is fragility
    "earnings_stability": +1,  # already inverted inside the calculation
    "asset_growth": -1,  # aggressive expansion tends to disappoint
}


def market_caps(
    panel: pd.DataFrame,
    fund: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.Series:
    """Market capitalisation per ISIN, using today's price and known share count.

    Share count comes from the most recent *visible* filing; price comes from
    ``as_of``. Pairing a stale price with a stale share count would measure the
    valuation of a company as it was months ago, not as the market prices it
    now.
    """
    latest = F.as_of(fund, as_of)
    if latest.empty:
        return pd.Series(dtype=float)

    day = panel[panel["date"] == as_of]
    if day.empty:
        # Fall back to the last session on or before as_of.
        prior = panel[panel["date"] <= as_of]
        if prior.empty:
            return pd.Series(dtype=float)
        day = prior[prior["date"] == prior["date"].max()]

    price = day.set_index("isin")["close"]
    shares = latest["shares_outstanding"]
    common = price.index.intersection(shares.index)
    mcap = price.loc[common] * shares.loc[common]
    return mcap[mcap > 0].rename("market_cap")


def compute(
    panel: pd.DataFrame,
    fund: pd.DataFrame,
    as_of: pd.Timestamp,
    isins: pd.Index | None = None,
) -> pd.DataFrame:
    """All fundamental factors for one formation date.

    Returns raw values; z-scoring happens in :mod:`nsefactor.factors` so that
    price and fundamental factors are standardised over the same cross-section.
    """
    latest = F.as_of(fund, as_of)
    if latest.empty:
        return pd.DataFrame()

    mcap = market_caps(panel, fund, as_of)
    ttm_pat = F.trailing_four_quarters(fund, as_of, "pat")
    ttm_rev = F.trailing_four_quarters(fund, as_of, "revenue")

    idx = latest.index if isins is None else pd.Index(isins)
    out = pd.DataFrame(index=idx)
    out["market_cap"] = mcap.reindex(idx)
    out["ttm_pat"] = ttm_pat.reindex(idx)
    out["ttm_revenue"] = ttm_rev.reindex(idx)
    out["net_worth"] = latest["net_worth"].reindex(idx)
    out["total_debt"] = latest["total_debt"].reindex(idx)

    # --- Value -----------------------------------------------------------
    # Earnings yield rather than P/E: the reciprocal is well behaved when
    # earnings approach zero, whereas P/E explodes and flips sign.
    out["earnings_yield"] = out["ttm_pat"] / out["market_cap"]

    # Book-to-price rather than price-to-book, for the same reason.
    # Negative net worth is a real state (accumulated losses) and must not be
    # read as extreme cheapness, so it is excluded rather than clipped.
    bp = out["net_worth"] / out["market_cap"]
    out["book_to_price"] = bp.where(out["net_worth"] > 0)

    # --- Quality ---------------------------------------------------------
    out["roe"] = (out["ttm_pat"] / out["net_worth"]).where(out["net_worth"] > 0)

    # Gross margin proxy: with only summary P&L items available, PAT margin is
    # what we can compute. Called gross_margin for continuity with the
    # literature but it is a net figure -- worth being explicit about.
    out["gross_margin"] = (out["ttm_pat"] / out["ttm_revenue"]).where(out["ttm_revenue"] > 0)

    out["debt_to_equity"] = (out["total_debt"] / out["net_worth"]).where(out["net_worth"] > 0)

    out["earnings_stability"] = _earnings_stability(fund, as_of).reindex(idx)
    out["asset_growth"] = _net_worth_growth(fund, as_of).reindex(idx)
    return out


def _earnings_stability(fund: pd.DataFrame, as_of: pd.Timestamp, n_quarters: int = 8) -> pd.Series:
    """Inverse coefficient of variation of quarterly earnings.

    Companies whose profits arrive predictably have historically outperformed
    those whose profits lurch, independent of how large those profits are.
    Scaling the standard deviation by mean absolute earnings makes the measure
    comparable across companies of very different size, and inverting it keeps
    the convention that higher is better.
    """
    visible = fund[fund["broadcast_date"] <= as_of]
    if visible.empty:
        return pd.Series(dtype=float)

    ordered = visible.sort_values(["isin", "period_end", "broadcast_date"])
    latest = ordered.groupby(["isin", "period_end"], as_index=False).tail(1)

    out = {}
    for isin, grp in latest.groupby("isin", sort=False):
        recent = grp.nlargest(n_quarters, "period_end")["pat"].dropna()
        if len(recent) < 6:
            continue
        scale = recent.abs().mean()
        if scale <= 0:
            continue
        cv = recent.std() / scale
        if not np.isfinite(cv):
            continue
        # Floor the coefficient of variation before inverting. Zero variance
        # would otherwise divide by zero, and treating it as unmeasurable would
        # score the *most* stable company as missing -- the exact opposite of
        # what this factor is for. Rounded or repeated reported figures make
        # near-zero variance a real occurrence, not just a synthetic one.
        out[isin] = 1.0 / max(cv, CV_FLOOR)
    return pd.Series(out, name="earnings_stability")


def _net_worth_growth(fund: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Year-over-year growth in net worth, as an asset-growth proxy.

    Rapid balance-sheet expansion has a well-documented tendency to precede
    disappointing returns -- capital raised in good times gets deployed at the
    margin. Signed negative in :data:`FUND_FACTOR_SIGNS`.
    """
    visible = fund[fund["broadcast_date"] <= as_of]
    if visible.empty:
        return pd.Series(dtype=float)

    ordered = visible.sort_values(["isin", "period_end", "broadcast_date"])
    latest = ordered.groupby(["isin", "period_end"], as_index=False).tail(1)

    out = {}
    for isin, grp in latest.groupby("isin", sort=False):
        g = grp.dropna(subset=["net_worth"]).nlargest(6, "period_end")
        if len(g) < 5:
            continue
        now = g.iloc[0]
        # Compare against the observation closest to four quarters earlier.
        target = now["period_end"] - pd.Timedelta(days=365)
        g_prior = g.iloc[1:].copy()
        g_prior["gap"] = (g_prior["period_end"] - target).abs().dt.days
        prior = g_prior.nsmallest(1, "gap").iloc[0]
        if prior["gap"] > 75 or prior["net_worth"] <= 0:
            continue
        out[isin] = now["net_worth"] / prior["net_worth"] - 1.0
    return pd.Series(out, name="asset_growth")
