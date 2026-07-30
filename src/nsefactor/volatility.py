"""Realised-volatility estimators, features, and forecast targets.

The pivot behind this module: monthly *returns* in this universe were close
enough to noise that a factor ranking built on them lost to an index fund.
Volatility is a different object. It clusters -- a turbulent month is
followed by a turbulent month far more often than chance -- which makes it
genuinely forecastable, and useful for the decisions a swing investor
actually makes: how much of a name to hold, and whether the regime changed.

Two design choices are carried over from the crypto version of this problem,
where both were learned the hard way:

**Predict a ratio, not a level.** One model trained across many instruments
with different baseline volatilities learns a compromise level that biases
every instrument. RELIANCE at 20% annualised and a smallcap at 60% cannot
share an intercept. The target here is ``log(future_vol / ewma_anchor)``, so
the model only has to predict *change* relative to a per-stock anchor.

**Anchor on EWMA, not trailing realised vol.** Trailing realised vol is
noisy, and dividing by a noisy denominator amplifies error into the target.
The EWMA estimate is smoother and makes a better anchor.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

YEAR = 252
# RiskMetrics' standard decay for daily data. 0.94 puts a ~30-day effective
# half-life on the estimate.
EWMA_LAMBDA = 0.94


def returns_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Dates x ISIN matrix of daily returns from the adjusted close."""
    prices = panel.pivot_table(index="date", columns="isin", values="adj_close", aggfunc="last")
    rets = prices.pct_change()
    # Residual junk that survives corporate-action adjustment. A single
    # session outside +-50% on a liquid name is a data artefact, and squaring
    # it would dominate any volatility estimate that contains it.
    return rets.where(rets.abs() <= 0.5)


def realized_vol(rets: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing annualised realised volatility. Causal by construction."""
    return rets.rolling(window, min_periods=max(3, window // 2)).std() * np.sqrt(YEAR)


def ewma_vol(rets: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    """RiskMetrics EWMA volatility.

    ``alpha = 1 - lam`` on squared returns, then annualised. Pandas' ``ewm``
    is causal: the value at t uses observations up to and including t.
    """
    var = (rets**2).ewm(alpha=1.0 - lam, min_periods=20).mean()
    return np.sqrt(var) * np.sqrt(YEAR)


def parkinson_vol(panel: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Range-based volatility from the high-low spread.

    Uses roughly five times less data than close-to-close for the same
    precision, because the intraday range carries information the close
    discards. Included as a feature, not a target.
    """
    high = panel.pivot_table(index="date", columns="isin", values="high", aggfunc="last")
    low = panel.pivot_table(index="date", columns="isin", values="low", aggfunc="last")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(high / low)
    log_hl = log_hl.where(np.isfinite(log_hl) & (log_hl >= 0))
    factor = 1.0 / (4.0 * np.log(2.0))
    var = (log_hl**2 * factor).rolling(window, min_periods=max(3, window // 2)).mean()
    return np.sqrt(var) * np.sqrt(YEAR)


def forward_vol(rets: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Realised volatility over the ``horizon`` days *after* each date.

    This is the prediction target and the only forward-looking quantity in
    the module. Everything else must be causal.
    """
    fwd = rets.shift(-horizon).rolling(horizon, min_periods=horizon // 2).std()
    return fwd.shift(0) * np.sqrt(YEAR)


def build_dataset(
    panel: pd.DataFrame,
    horizon: int = 21,
    universe_isins: pd.Index | None = None,
    require_target: bool = True,
) -> pd.DataFrame:
    """Long-format feature/target table for volatility forecasting.

    One row per (date, isin). Features are all trailing; ``target_log_ratio``
    is ``log(forward_vol / ewma_anchor)`` and is the quantity models predict.

    With ``require_target=False``, rows whose forward window extends past the
    end of the data are kept, with a null target. Those rows are exactly the
    most recent dates -- including today -- which is when a forecast is
    actually wanted. Dropping them silently leaves the live shortlist with no
    risk estimate at all.
    """
    if universe_isins is not None:
        panel = panel[panel["isin"].isin(universe_isins)]

    rets = returns_wide(panel)

    feats = {
        "rv_5": realized_vol(rets, 5),
        "rv_21": realized_vol(rets, 21),
        "rv_63": realized_vol(rets, 63),
        "rv_126": realized_vol(rets, 126),
        "ewma": ewma_vol(rets),
        "parkinson_21": parkinson_vol(panel, 21),
    }

    # Trailing return and absolute return: volatility rises after drawdowns
    # more than after rallies, so the sign carries information.
    prices = panel.pivot_table(index="date", columns="isin", values="adj_close", aggfunc="last")
    feats["ret_21"] = prices.pct_change(21)
    feats["ret_5"] = prices.pct_change(5)

    # Market-wide volatility: the cross-sectional median, which captures the
    # regime every stock sits inside.
    market = feats["rv_21"].median(axis=1)
    feats["market_rv_21"] = pd.DataFrame(
        np.repeat(market.to_numpy()[:, None], rets.shape[1], axis=1),
        index=rets.index,
        columns=rets.columns,
    )

    # Ratios: where does this stock sit against its own longer-run vol and
    # against the market? Scale-free, so they transfer across stocks.
    feats["rv_ratio_5_63"] = feats["rv_5"] / feats["rv_63"]
    feats["rv_ratio_21_126"] = feats["rv_21"] / feats["rv_126"]
    feats["rv_vs_market"] = feats["rv_21"] / feats["market_rv_21"]

    turnover = panel.pivot_table(index="date", columns="isin", values="turnover", aggfunc="last")
    feats["turnover_ratio"] = (
        turnover.rolling(5, min_periods=3).mean() / turnover.rolling(63, min_periods=30).mean()
    )

    target = forward_vol(rets, horizon)
    anchor = feats["ewma"]

    frames = []
    for name, wide in feats.items():
        frames.append(wide.stack(future_stack=True).rename(name))
    df = pd.concat(frames, axis=1)
    df["forward_vol"] = target.stack(future_stack=True)
    df["anchor"] = anchor.stack(future_stack=True)

    df = df.reset_index().rename(columns={"level_0": "date", "level_1": "isin"})
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "isin"})

    # The anchor must be usable as a denominator. Without it there is nothing
    # to scale a forecast against, so those rows go regardless.
    ok = df["anchor"].notna() & (df["anchor"] > 1e-6)
    if require_target:
        ok &= df["forward_vol"].notna() & (df["forward_vol"] > 1e-6)
    df = df[ok].copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = df["forward_vol"] / df["anchor"]
        df["target_log_ratio"] = np.log(ratio.where(ratio > 0))
    return df


FEATURE_COLUMNS = [
    "rv_5",
    "rv_21",
    "rv_63",
    "rv_126",
    "ewma",
    "parkinson_21",
    "ret_21",
    "ret_5",
    "market_rv_21",
    "rv_ratio_5_63",
    "rv_ratio_21_126",
    "rv_vs_market",
    "turnover_ratio",
]


# ---------------------------------------------------------------------------
# Baselines. Each returns a forecast of annualised vol over the next horizon.
# ---------------------------------------------------------------------------


def baseline_persistence(df: pd.DataFrame) -> pd.Series:
    """Tomorrow looks like today: trailing 21-day realised vol, unchanged.

    The bar every model must clear. For hourly crypto *returns* this baseline
    was unbeatable, which is exactly why it is here.
    """
    return df["rv_21"]


def baseline_ewma(df: pd.DataFrame) -> pd.Series:
    """RiskMetrics EWMA, i.e. predict the anchor itself (log ratio of zero)."""
    return df["anchor"]


def baseline_garch(
    panel: pd.DataFrame,
    eval_rows: pd.DataFrame,
    horizon: int = 21,
    min_obs: int = 500,
    max_stocks: int | None = None,
) -> pd.Series:
    """GARCH(1,1) forecast, fitted per stock on data up to each eval date.

    Genuinely expensive: one model fit per (stock, date). Refits are done
    once per stock per evaluation month rather than per day, and the caller
    can cap the number of stocks. Returns NaN where a fit fails to converge.
    """
    from arch import arch_model

    rets = returns_wide(panel)
    out = pd.Series(index=eval_rows.index, dtype=float)

    isins = eval_rows["isin"].unique()
    if max_stocks is not None:
        isins = isins[:max_stocks]

    for isin in isins:
        if isin not in rets.columns:
            continue
        series = rets[isin].dropna() * 100.0  # arch prefers percentage returns
        rows = eval_rows[eval_rows["isin"] == isin]
        for idx, row in rows.iterrows():
            hist = series[series.index <= row["date"]]
            if len(hist) < min_obs:
                continue
            try:
                fit = arch_model(hist, vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
                fc = fit.forecast(horizon=horizon, reindex=False)
                daily_var = float(fc.variance.to_numpy().mean())
                out.loc[idx] = np.sqrt(daily_var) / 100.0 * np.sqrt(YEAR)
            except Exception:
                continue
    return out
