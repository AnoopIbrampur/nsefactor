"""Recover corporate-action adjustment factors from the bhavcopy itself.

NSE does not publish an adjusted price series, and its corporate-actions feed
sits behind the cookie-gated www host. We do not need either.

On the ex-date of a split, bonus, or large dividend, NSE reports
``prevclose`` as the *adjusted* prior close while the previous session's
``close`` is the raw, unadjusted figure. Their ratio is therefore the
adjustment factor for that action:

    factor = prevclose(t) / close(t-1)

A 1:2 split shows up as ~0.5, a 1:1 bonus as ~0.5, a 5% dividend as ~0.95.
Chaining these backwards from the present gives a total-return-ish adjusted
close built only from data we already have on disk.

This matters more than it sounds. On an unadjusted series a 1:5 split reads
as an -80% single-day return, which any momentum factor will happily rank as
the worst stock in the market on exactly the day nothing bad happened.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, Config

log = logging.getLogger(__name__)


def detect_factors(panel: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Return per-(isin, date) adjustment factors where an action is implied.

    Only rows whose factor departs from 1.0 by more than the configured
    tolerance are returned; everything else is an ordinary session.
    """
    df = panel.sort_values(["isin", "date"])[["isin", "date", "close", "prevclose"]].copy()
    df["prev_close_actual"] = df.groupby("isin", sort=False)["close"].shift(1)

    valid = df["prev_close_actual"].notna() & (df["prev_close_actual"] > 0)
    df = df.loc[valid].copy()
    df["factor"] = df["prevclose"] / df["prev_close_actual"]

    lo, hi = cfg.ca_factor_bounds
    implausible = (df["factor"] < lo) | (df["factor"] > hi)
    if implausible.any():
        log.warning(
            "ignoring %d implausible adjustment factors (outside %.2f-%.2f)",
            int(implausible.sum()),
            lo,
            hi,
        )
    df.loc[implausible, "factor"] = 1.0

    moved = (df["factor"] - 1.0).abs() > cfg.ca_detect_tolerance
    actions = df.loc[moved, ["isin", "date", "factor"]].reset_index(drop=True)
    log.info("detected %d corporate actions across %d ISINs", len(actions), actions["isin"].nunique())
    return actions


def adjusted_close(panel: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Add ``adj_close`` and ``adj_factor`` columns to ``panel``.

    The series is normalised so the most recent close of each ISIN equals its
    raw close, i.e. adjustments propagate backwards in time. That is the
    convention every retail data source uses, and it keeps today's prices
    recognisable when the uncle looks at them.
    """
    df = panel.sort_values(["isin", "date"]).copy()
    actions = detect_factors(df, cfg)

    if actions.empty:
        df["adj_factor"] = 1.0
        df["adj_close"] = df["close"]
        return df

    key = pd.MultiIndex.from_frame(actions[["isin", "date"]])
    per_day = pd.Series(actions["factor"].to_numpy(), index=key)

    idx = pd.MultiIndex.from_frame(df[["isin", "date"]])
    day_factor = per_day.reindex(idx).fillna(1.0).to_numpy()
    df["_day_factor"] = day_factor

    # Cumulative product of all *future* factors, per ISIN. A price on day t
    # must be scaled by every action that happens after t.
    def _backward_cum(group: pd.Series) -> pd.Series:
        rev = group.iloc[::-1]
        # shift(1) so a day's own factor does not adjust its own close --
        # the ex-date close is already quoted post-action.
        return rev.shift(1).fillna(1.0).cumprod().iloc[::-1]

    df["adj_factor"] = (
        df.groupby("isin", sort=False)["_day_factor"].transform(_backward_cum)
    )
    df["adj_close"] = df["close"] * df["adj_factor"]
    return df.drop(columns="_day_factor")


def adjusted_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from the adjusted close, per ISIN."""
    df = panel.sort_values(["isin", "date"]).copy()
    df["ret"] = df.groupby("isin", sort=False)["adj_close"].pct_change()
    # Guard against the residual junk that survives adjustment: a return
    # outside +-50% in one session on a liquid name is almost always a data
    # artefact, not a price move.
    df.loc[df["ret"].abs() > 0.5, "ret"] = np.nan
    return df
