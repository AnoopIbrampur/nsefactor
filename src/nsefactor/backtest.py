"""Walk-forward backtest for a monthly-rebalanced, equal-weighted long book.

Timing convention, which is the part most backtests get quietly wrong:

* Factors are formed from bars up to and including the rebalance date ``t``.
* The portfolio is entered at the **close of t+1**, one full session later.
* It is held until the next rebalance date, then rolled.

The one-day lag is not conservatism for its own sake. Forming a signal on
t's close and also transacting at t's close assumes you could compute the
signal and trade simultaneously at the closing bell, which nobody can. Under
that assumption a mediocre strategy can look excellent.

Delisting is handled explicitly rather than by omission. If a held name stops
trading mid-period, the position is carried to its last observed price and
exited there. Dropping it instead would silently discard exactly the losses
that make survivorship bias so flattering.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, Config

log = logging.getLogger(__name__)


def month_end_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last actual trading session of each calendar month."""
    s = pd.Series(trading_days, index=trading_days)
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).last().to_numpy())


def forward_returns(
    panel: pd.DataFrame,
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
    isins: pd.Index,
) -> pd.Series:
    """Realised return per ISIN from ``entry`` close to ``exit_`` close.

    A name that stops trading between the two dates is exited at its last
    observed close inside the window, not dropped. A name with no bar at
    ``entry`` was not buyable and returns NaN so the caller can exclude it
    before weights are set.
    """
    window = panel[(panel["date"] >= entry) & (panel["date"] <= exit_)]
    window = window[window["isin"].isin(isins)]
    if window.empty:
        return pd.Series(dtype=float, index=isins)

    wide = window.pivot_table(index="date", columns="isin", values="adj_close", aggfunc="last")
    if entry not in wide.index:
        return pd.Series(dtype=float, index=isins)

    start = wide.loc[entry]
    # Last non-null price in the window: the true exit for a name that stops
    # trading partway through.
    end = wide.ffill().iloc[-1]

    out = (end / start) - 1.0
    out = out.where(start.notna() & (start > 0) & end.notna())
    return out.reindex(isins)


def _weights_from_scores(scores: pd.Series, n_holdings: int) -> pd.Series:
    """Equal weight across the top ``n_holdings`` names by score."""
    top = scores.dropna().nlargest(n_holdings)
    if top.empty:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(top), index=top.index)


def run(
    panel: pd.DataFrame,
    score_fn,
    trading_days: pd.DatetimeIndex,
    cfg: Config = DEFAULT_CONFIG,
    start: str | pd.Timestamp | None = None,
    warmup_days: int = 300,
) -> dict:
    """Execute the walk-forward backtest.

    ``score_fn(panel, as_of) -> pd.Series`` returns a score per ISIN, higher
    being better. It must look only at bars dated on or before ``as_of``;
    :mod:`nsefactor.factors` enforces this and the test suite checks it.

    Returns a dict with the period-by-period ledger, the daily equity curve,
    and realised turnover.
    """
    rebals = month_end_dates(trading_days)
    # Skip the head of the sample: factors need a year of history behind them.
    if len(trading_days) > warmup_days:
        earliest = trading_days[warmup_days]
        rebals = rebals[rebals >= earliest]
    if start is not None:
        rebals = rebals[rebals >= pd.Timestamp(start)]

    day_pos = {d: i for i, d in enumerate(trading_days)}
    periods = []
    prev_weights = pd.Series(dtype=float)

    for i, form_date in enumerate(rebals[:-1]):
        # Enter one session after formation.
        idx = day_pos[form_date]
        if idx + 1 >= len(trading_days):
            break
        entry = trading_days[idx + 1]
        exit_ = rebals[i + 1]
        if entry >= exit_:
            continue

        scores = score_fn(panel, form_date)
        if scores is None or scores.dropna().empty:
            log.debug("no scores at %s", form_date.date())
            continue

        weights = _weights_from_scores(scores, cfg.n_holdings)
        if weights.empty:
            continue

        rets = forward_returns(panel, entry, exit_, weights.index)
        # A name with no entry bar could not have been bought; drop it and
        # re-spread the weight across what was actually purchasable.
        tradable = rets.notna()
        if not tradable.any():
            continue
        weights = weights[tradable]
        weights = weights / weights.sum()
        rets = rets[tradable]

        gross = float((weights * rets).sum())

        # Turnover is the total absolute weight change against the book we
        # were already holding, charged at the per-side rate.
        aligned_prev = prev_weights.reindex(weights.index).fillna(0.0)
        dropped = prev_weights.drop(weights.index, errors="ignore")
        turnover = float(np.abs(weights - aligned_prev).sum() + np.abs(dropped).sum()) / 2.0
        cost = turnover * 2.0 * cfg.cost_bps_per_side / 1e4

        periods.append(
            {
                "form_date": form_date,
                "entry": entry,
                "exit": exit_,
                "n_holdings": int(len(weights)),
                "gross_return": gross,
                "turnover": turnover,
                "cost": cost,
                "net_return": gross - cost,
            }
        )
        prev_weights = weights

    ledger = pd.DataFrame(periods)
    if ledger.empty:
        return {"ledger": ledger, "equity": pd.Series(dtype=float)}

    ledger["gross_equity"] = (1.0 + ledger["gross_return"]).cumprod()
    ledger["net_equity"] = (1.0 + ledger["net_return"]).cumprod()
    return {
        "ledger": ledger,
        "equity": ledger.set_index("exit")["net_equity"],
        "gross_equity": ledger.set_index("exit")["gross_equity"],
    }


def equal_weight_universe(
    panel: pd.DataFrame,
    universe_fn,
    trading_days: pd.DatetimeIndex,
    cfg: Config = DEFAULT_CONFIG,
    **kwargs,
) -> dict:
    """Benchmark: hold the whole investable universe, equally weighted.

    This is the honest comparison for a stock-picking strategy. Beating the
    Nifty 50 could just mean the universe tilted small-cap during a small-cap
    rally; beating an equal-weighted version of *the same universe* means the
    ranking actually selected.
    """

    def score(p, as_of):
        sel = universe_fn(p, as_of)
        if sel is None or len(sel) == 0:
            return None
        return pd.Series(1.0, index=sel.index)

    wide_cfg = Config(**{**cfg.__dict__, "n_holdings": cfg.universe_size})
    return run(panel, score, trading_days, wide_cfg, **kwargs)
