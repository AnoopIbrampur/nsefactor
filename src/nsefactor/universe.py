"""Point-in-time investable universe, reconstructed from the bhavcopy.

The obvious way to build an Indian equity universe is to download today's
Nifty 500 constituent list and use it for the whole backtest. That is also
the single most effective way to fake a good result. Today's list contains
only companies that survived and stayed large enough to remain in the index;
every name that was delisted, acquired, or collapsed has been quietly
removed. A backtest run on it earns returns that were unavailable to anyone
holding the portfolio at the time.

We avoid the problem instead of correcting for it. A bhavcopy dated 2016-03-31
contains exactly the stocks that traded on 2016-03-31 -- including the ones
that no longer exist. Ranking *that* file by liquidity reproduces an
investable universe as it stood, with no forward-looking information and no
external index-membership feed to source.

The resulting universe is not the Nifty 500. It is a liquidity-ranked top-N,
which is what the index approximately is anyway (see :func:`overlap_with_index`
for the measured agreement).
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import ARCHIVE_HOST, DEFAULT_CONFIG, USER_AGENT, Config

log = logging.getLogger(__name__)

NIFTY500_URL = f"{ARCHIVE_HOST}/content/indices/ind_nifty500list.csv"


def formation_stats(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 126,
    cfg: Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Per-ISIN liquidity statistics over the window *ending* at ``as_of``.

    The window is strictly historical: ``as_of`` is included, nothing after
    it is. Every downstream selection therefore uses only information a
    person standing on ``as_of`` could have had.
    """
    hist = panel[panel["date"] <= as_of]
    if hist.empty:
        return pd.DataFrame()

    days = pd.DatetimeIndex(sorted(hist["date"].unique()))[-lookback_days:]
    if len(days) == 0:
        return pd.DataFrame()
    window = hist[hist["date"].isin(days)]

    stats = window.groupby("isin").agg(
        symbol=("symbol", "last"),
        median_turnover=("turnover", "median"),
        n_days=("date", "nunique"),
        last_close=("close", "last"),
    )
    stats["trading_frac"] = stats["n_days"] / len(days)
    stats["window_days"] = len(days)
    return stats


def select(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 126,
    cfg: Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """The investable universe as of ``as_of``, most liquid first.

    Filters, in order: traded on ``as_of`` at all, traded on enough of the
    formation window to be rankable, cleared the turnover floor. Survivors
    are ranked by median turnover and cut at ``cfg.universe_size``.
    """
    stats = formation_stats(panel, as_of, lookback_days, cfg)
    if stats.empty:
        return stats

    # Must be trading on the selection date itself -- a stock suspended into
    # the rebalance is not buyable no matter how liquid it used to be.
    live = set(panel.loc[panel["date"] == as_of, "isin"])
    stats = stats[stats.index.isin(live)]

    eligible = stats[
        (stats["trading_frac"] >= cfg.min_trading_days_frac)
        & (stats["median_turnover"] >= cfg.min_median_turnover)
    ]
    return eligible.nlargest(cfg.universe_size, "median_turnover")


def build_history(
    panel: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    lookback_days: int = 126,
    cfg: Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Long-format universe membership across every rebalance date."""
    rows = []
    for dt in rebalance_dates:
        sel = select(panel, dt, lookback_days, cfg)
        if sel.empty:
            log.warning("empty universe at %s", dt.date())
            continue
        rows.append(
            pd.DataFrame(
                {
                    "date": dt,
                    "isin": sel.index,
                    "symbol": sel["symbol"].to_numpy(),
                    "median_turnover": sel["median_turnover"].to_numpy(),
                    "rank": range(1, len(sel) + 1),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["date", "isin", "symbol", "median_turnover", "rank"])
    return pd.concat(rows, ignore_index=True)


def turnover_rate(history: pd.DataFrame) -> pd.Series:
    """Fraction of the universe replaced at each rebalance.

    A sane liquidity-ranked universe churns a few percent a month. A large
    spike means the liquidity filter is chattering around its threshold, not
    that the market changed.
    """
    out = {}
    prev: set | None = None
    for dt, grp in history.groupby("date"):
        cur = set(grp["isin"])
        if prev is not None:
            out[dt] = len(cur - prev) / len(cur)
        prev = cur
    return pd.Series(out, name="universe_turnover")


def fetch_current_index(url: str = NIFTY500_URL) -> pd.DataFrame:
    """Today's published Nifty 500 list. Validation only -- never for selection."""
    import io

    import requests

    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={"Company Name": "name", "Symbol": "symbol", "ISIN Code": "isin"})


def overlap_with_index(selected: pd.DataFrame, index_list: pd.DataFrame) -> float:
    """Agreement between our reconstructed universe and the real index today.

    Sanity check on the liquidity proxy. Perfect agreement is not expected or
    wanted -- the index applies float, sector, and listing-history rules we
    deliberately do not model -- but a low number would mean the proxy is
    picking a different market than the one the uncle trades.
    """
    ours = set(selected.index if selected.index.name == "isin" else selected["isin"])
    theirs = set(index_list["isin"].str.strip())
    if not theirs:
        return float("nan")
    return len(ours & theirs) / len(theirs)
