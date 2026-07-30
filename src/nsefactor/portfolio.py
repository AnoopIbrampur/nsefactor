"""Risk-aware portfolio construction.

The factor ranking in :mod:`nsefactor.factors` had real signal -- monotonic
deciles, composite IC t = 4.01 -- and still lost to an index fund. The
ranking was not the problem. Construction was:

* **20 names is too few.** Idiosyncratic noise swamped a weak signal.
* **Equal weighting ignored risk.** The book ran 22% volatility against the
  index's 14.5%, almost all of it an uncontrolled tilt toward small, jumpy
  names that happen to score well on price factors.
* **Turnover was unmanaged**, costing 2-6%/yr depending on the factor set.

Each fix below targets one of those, and none of them touch the signal. The
volatility forecast from :mod:`nsefactor.volatility` supplies the risk
estimate for sizing, which is what that model is actually for.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def inverse_vol_weights(
    chosen: pd.Index,
    risk: pd.Series,
    max_weight: float = 0.08,
    floor_vol: float = 0.10,
) -> pd.Series:
    """Weights proportional to 1/forecast volatility, capped.

    A calm large-cap and a jumpy micro-cap contribute wildly different
    amounts of risk per rupee. Equal weighting therefore hands most of the
    portfolio's actual risk to its most volatile handful of names. Sizing on
    1/vol equalises risk contribution instead of rupee contribution.

    ``max_weight`` stops the calmest name dominating; ``floor_vol`` stops a
    stock with an implausibly low vol estimate from doing the same.
    """
    n = len(chosen)
    if n == 0:
        return pd.Series(dtype=float)

    r = risk.reindex(chosen)
    # Fall back to the cross-sectional median where a forecast is missing,
    # so a name is never sized as if it were riskless.
    r = r.fillna(r.median()).clip(lower=floor_vol)
    if r.isna().all():
        return pd.Series(1.0 / n, index=chosen)

    # A cap below 1/n cannot be satisfied by any book that sums to one, so
    # respect the binding constraint rather than oscillating against it.
    cap = max(max_weight, 1.0 / n)

    w = 1.0 / r
    w = w / w.sum()

    # Water-filling: once a name is at the cap it stays locked, and the
    # surplus is redistributed only among names that have never been capped.
    # Redistributing into already-capped names is what makes the naive loop
    # oscillate and finish above the cap.
    locked = pd.Series(False, index=chosen)
    for _ in range(n + 1):
        over = (w > cap + 1e-12) & ~locked
        if not over.any():
            break
        locked |= over
        w[locked] = cap
        free = ~locked
        remaining = 1.0 - float(w[locked].sum())
        if not free.any() or remaining <= 0:
            break
        w[free] = remaining * w[free] / w[free].sum()
    return w / w.sum()


def cap_by_group(weights: pd.Series, groups: pd.Series, max_group: float = 0.30) -> pd.Series:
    """Scale down any group (e.g. industry) exceeding ``max_group`` of the book.

    Price factors cluster by sector -- when metals run, every metal name looks
    like a winner at once -- so an unconstrained book can quietly become a
    single sector bet.
    """
    w = weights.copy()
    g = groups.reindex(w.index)

    n_groups = g.nunique(dropna=False)
    if n_groups * max_group < 1.0 - 1e-12:
        log.warning(
            "group cap %.2f is infeasible across %d groups; using %.4f",
            max_group,
            n_groups,
            1.0 / n_groups,
        )
        max_group = 1.0 / n_groups

    # Same water-filling as the weight cap, for the same reason: a group that
    # has already been pushed down to its cap must not receive redistributed
    # weight, or capping one group re-inflates another in a cycle.
    locked: set = set()
    for _ in range(n_groups + 1):
        totals = w.groupby(g).sum()
        breached = [name for name, t in totals.items() if t > max_group + 1e-12 and name not in locked]
        if not breached:
            break
        for name in breached:
            members = g[g == name].index
            total = float(w[members].sum())
            if total > 0:
                w[members] *= max_group / total
            locked.add(name)

        free_mask = ~g.isin(locked)
        remaining = 1.0 - float(w[~free_mask].sum())
        if not free_mask.any() or remaining <= 0:
            break
        free_total = float(w[free_mask].sum())
        if free_total > 0:
            w[free_mask] = remaining * w[free_mask] / free_total
    return w / w.sum()


def select_with_buffer(
    scores: pd.Series,
    n_holdings: int,
    held: pd.Index | None = None,
    buffer_mult: float = 2.0,
) -> pd.Index:
    """Top ``n_holdings`` by score, retaining held names inside a wider band."""
    ranked = scores.dropna().sort_values(ascending=False)
    if ranked.empty:
        return pd.Index([])
    if held is None or buffer_mult <= 1.0:
        return ranked.index[:n_holdings]

    band = set(ranked.index[: int(n_holdings * buffer_mult)])
    keep = [i for i in ranked.index if i in band and i in set(held)][:n_holdings]
    room = n_holdings - len(keep)
    keep_set = set(keep)
    fill = [i for i in ranked.index[:n_holdings] if i not in keep_set][:room]
    return pd.Index(keep + fill)


def build(
    scores: pd.Series,
    risk: pd.Series | None,
    n_holdings: int,
    held: pd.Index | None = None,
    buffer_mult: float = 2.0,
    max_weight: float = 0.08,
    groups: pd.Series | None = None,
    max_group: float = 0.30,
) -> pd.Series:
    """Select names by score, then size them by forecast risk."""
    chosen = select_with_buffer(scores, n_holdings, held, buffer_mult)
    if len(chosen) == 0:
        return pd.Series(dtype=float)

    if risk is None:
        w = pd.Series(1.0 / len(chosen), index=chosen)
    else:
        w = inverse_vol_weights(chosen, risk, max_weight=max_weight)

    if groups is not None:
        w = cap_by_group(w, groups, max_group=max_group)
    return w
