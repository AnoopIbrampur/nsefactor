"""Risk-aware stock shortlist: construction fixes on a signal we already have.

Usage:
    python scripts/shortlist.py            # backtest the construction, then print today's list
    python scripts/shortlist.py --today    # skip the backtest, just print today's list

The factor ranking had real signal and still lost to an index fund. This
script does not touch the signal. It fixes the three construction faults the
evaluation identified:

  1. 20 holdings was too concentrated for a weak signal  -> more names
  2. equal weighting let jumpy small-caps carry the risk -> inverse-vol sizing
  3. turnover cost 2-6%/yr                               -> rank buffer

The risk estimate comes from the volatility model, trained only on data
before the test period. That model is the one part of this repo that clearly
beat its baselines, and position sizing is what it is for.

**A note on evidence.** The test period has already been examined once, when
the equal-weighted baseline was evaluated. A second pass over the same held-out
years is weaker evidence than the first, however carefully the parameters were
tuned on train. Treat the numbers here as suggestive, not as a fresh
out-of-sample result. The forward test in the roadmap is the honest fix.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from nsefactor import (
    adjust,
    backtest,
    benchmark,
    factors,
    metrics,
    portfolio,
    universe,
    volatility as volmod,
)
from nsefactor.config import DATA_DIR, DEFAULT_CONFIG

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

CFG = DEFAULT_CONFIG
TRAIN_END = pd.Timestamp("2020-12-31")
# Fixed a priori, not re-selected here. Momentum was the factor that held up
# out of sample (t rose from 1.98 to 2.31); low-vol is the other one with
# strong published support. Reversal and illiquidity were dropped on training
# evidence and stay dropped.
FACTOR_SET = ("mom_12_1", "vol_126")
HORIZON = 21


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def train_risk_model(voldf: pd.DataFrame, train_end: pd.Timestamp):
    """Fit the volatility model on data strictly before ``train_end``."""
    gap = pd.Timedelta(days=int(HORIZON * 1.6))
    tr = voldf[voldf["date"] <= train_end - gap]
    X = tr[volmod.FEATURE_COLUMNS].to_numpy(dtype=float)
    y = tr["target_log_ratio"].to_numpy(dtype=float)
    model = HistGradientBoostingRegressor(
        max_iter=264,  # chosen on the validation slice in scripts/vol_model.py
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=CFG.seed,
    )
    model.fit(X, y)
    return model


def risk_forecasts(model, voldf: pd.DataFrame) -> pd.Series:
    """Forecast annualised vol for every (date, isin) row, indexed by both."""
    X = voldf[volmod.FEATURE_COLUMNS].to_numpy(dtype=float)
    pred_ratio = model.predict(X)
    fc = voldf["anchor"].to_numpy() * np.exp(pred_ratio)
    return pd.Series(fc, index=pd.MultiIndex.from_arrays([voldf["date"], voldf["isin"]]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", action="store_true", help="skip backtest, print current list")
    ap.add_argument("--n", type=int, default=None, help="override holdings count")
    args = ap.parse_args()

    raw = pd.read_parquet(DATA_DIR / "bhavcopy.parquet")
    # Bridge ISIN changes before adjusting, so a company that switched ISIN
    # keeps one continuous history instead of looking like a recent listing.
    raw = adjust.apply_isin_links(raw)
    panel = adjust.adjusted_close(raw)
    days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rebals = backtest.month_end_dates(days)
    rebals = rebals[rebals >= days[300]]

    print(f"panel {len(panel):,} rows | {len(rebals)} rebalances "
          f"{rebals[0].date()} -> {rebals[-1].date()}")

    # ---- Factor scores at every rebalance -------------------------------
    print("computing factor scores...")
    fcache: dict = {}
    for dt in rebals:
        sel = universe.select(panel, dt, cfg=CFG)
        if sel.empty:
            fcache[dt] = None
            continue
        f = factors.compute_all(panel, dt, sel.index, use=FACTOR_SET)
        fcache[dt] = f

    # ---- Risk model ------------------------------------------------------
    print("building volatility dataset and fitting the risk model...")
    members = sorted({i for f in fcache.values() if f is not None for i in f.index})

    # Built once, with rows that have no realised target retained. Those rows
    # are the most recent dates -- including the one we want a list for today.
    # Training filters them out; prediction must not.
    voldf = volmod.build_dataset(
        panel, horizon=HORIZON, universe_isins=pd.Index(members), require_target=False
    )
    voldf = voldf.dropna(subset=volmod.FEATURE_COLUMNS)

    trainable = voldf.dropna(subset=["target_log_ratio"])
    model = train_risk_model(trainable, TRAIN_END)

    risk = risk_forecasts(model, voldf)
    latest_risk_date = voldf["date"].max()
    print(f"risk forecasts for {len(risk):,} (date, stock) pairs, "
          f"latest {latest_risk_date.date()}")

    def make_weight_fn(n_holdings: int, max_weight: float, buffer_mult: float):
        def weight_fn(scores, as_of, held):
            try:
                r = risk.xs(as_of, level=0)
            except KeyError:
                r = pd.Series(dtype=float)
            return portfolio.build(
                scores,
                r if len(r) else None,
                n_holdings=n_holdings,
                held=held,
                buffer_mult=buffer_mult,
                max_weight=max_weight,
            )

        return weight_fn

    def score_fn(p, as_of):
        f = fcache.get(as_of)
        return None if f is None else f["composite"]

    if not args.today:
        # ---- Tune construction on the training period only --------------
        section("CONSTRUCTION GRID (training period only, 2016-2020)")
        print("Signal is fixed. Only holdings count, weight cap, and buffer vary.\n")
        grid = []
        for n in (20, 40, 60):
            for mw in (0.05, 0.08):
                r = backtest.run(
                    panel, score_fn, days, CFG, end=TRAIN_END,
                    weight_fn=make_weight_fn(n, mw, 2.0),
                )
                led = r["ledger"]
                if led.empty:
                    continue
                grid.append({
                    "n": n, "max_w": mw,
                    "CAGR%": round(metrics.cagr(led["net_return"]) * 100, 2),
                    "Vol%": round(metrics.volatility(led["net_return"]) * 100, 2),
                    "Sharpe": round(metrics.sharpe(led["net_return"]), 2),
                    "MaxDD%": round(metrics.max_drawdown(led["net_return"]) * 100, 2),
                    "turn%": round(led["turnover"].mean() * 100, 1),
                })
        g = pd.DataFrame(grid)
        print(g.to_string(index=False))
        best = g.loc[g["Sharpe"].idxmax()]
        n_best, mw_best = int(best["n"]), float(best["max_w"])
        print(f"\nbest on train by Sharpe: n={n_best}, max_weight={mw_best}")

        # ---- Single evaluation on the held-out years --------------------
        section("TEST PERIOD (2021+)")
        print("Equal-weighted top-20 is the version that lost to the index.")
        print("Everything else differs only in construction.\n")

        rows = []
        res = backtest.run(panel, score_fn, days, CFG, start=TRAIN_END,
                           weight_fn=make_weight_fn(n_best, mw_best, 2.0))
        led = res["ledger"]
        rows.append(metrics.summary(led["net_return"], f"risk-weighted top-{n_best} (net)"))

        old = backtest.run(panel, score_fn, days, CFG, start=TRAIN_END, buffer_mult=1.0)
        if not old["ledger"].empty:
            rows.append(metrics.summary(old["ledger"]["net_return"], "equal-weight top-20 (net)"))

        eqw = backtest.equal_weight_universe(
            panel, lambda p, d: universe.select(p, d, cfg=CFG), days, CFG, start=TRAIN_END
        )
        if not eqw["ledger"].empty:
            rows.append(metrics.summary(eqw["ledger"]["net_return"], "equal-weight universe (net)"))

        exits = pd.DatetimeIndex(led["exit"])
        try:
            idx = pd.read_parquet(DATA_DIR / "indices.parquet")
            for nm in ("Nifty 50", "Nifty 500"):
                try:
                    s = benchmark.series(idx, nm)
                except KeyError:
                    continue
                rows.append(metrics.summary(
                    metrics.align_monthly(benchmark.total_return_proxy(s).dropna(), exits),
                    f"{nm} (total-return approx)",
                ))
        except FileNotFoundError:
            pass

        print(metrics.compare(rows).to_string())
        print(f"\nturnover {led['turnover'].mean():.1%}/mo, "
              f"cost drag {led['cost'].mean() * 12 * 100:.2f}%/yr")
        args.n = args.n or n_best
    else:
        args.n = args.n or 40

    # ---- Today's shortlist ----------------------------------------------
    section("CURRENT SHORTLIST")
    as_of = rebals[-1]
    f = fcache[as_of]
    sel = universe.select(panel, as_of, cfg=CFG)

    try:
        r = risk.xs(as_of, level=0)
    except KeyError:
        r = pd.Series(dtype=float)

    # Sector labels from the published index list. Only current labels are
    # available, which is why the cap is applied to the live shortlist and not
    # to the backtest -- using today's sectors historically would be a mild
    # lookahead, and industry classification is not in the bhavcopy at all.
    sectors = None
    try:
        idx_list = universe.fetch_current_index()
        by_isin = idx_list.set_index(idx_list["isin"].str.strip())["Industry"]
        sectors = pd.Series(sel.index.map(by_isin), index=sel.index)
        # Our universe is a liquidity-ranked top 500, which overlaps the
        # published Nifty 500 by ~81%, so some names carry no industry label.
        # Pooling them under one "Unknown" bucket would cap a handful of
        # unrelated companies as though they were a sector. Each unlabelled
        # name becomes its own group so the cap simply does not bind on it.
        missing = sectors.isna()
        sectors[missing] = ["unclassified:" + str(i) for i in sectors.index[missing]]
    except Exception as exc:
        print(f"(could not fetch sector labels: {exc}; sector cap skipped)")

    weights = portfolio.build(
        f["composite"],
        r if len(r) else None,
        n_holdings=args.n,
        buffer_mult=1.0,
        max_weight=0.08,
        groups=sectors,
        max_group=0.25,
    )

    out = pd.DataFrame({
        "symbol": sel["symbol"].reindex(weights.index),
        "sector": (sectors.reindex(weights.index) if sectors is not None else "n/a"),
        "weight%": (weights * 100).round(2),
        "price": sel["last_close"].reindex(weights.index).round(2),
        "fcast_vol%": (r.reindex(weights.index) * 100).round(1),
        "mom_12_1%": (f["mom_12_1"].reindex(weights.index) * 100).round(1),
        "vol_126%": (f["vol_126"].reindex(weights.index) * 100).round(1),
        "score": f["composite"].reindex(weights.index).round(3),
    }).sort_values("score", ascending=False)

    print(f"as of {as_of.date()}  |  {len(out)} names  |  "
          f"universe {len(sel)}  |  weights sum {weights.sum():.3f}\n")
    print(out.to_string(index=False))

    if sectors is not None:
        print("\nsector exposure (capped at 25%):")
        exposure = (weights.groupby(sectors.reindex(weights.index)).sum() * 100).round(2)
        print(exposure.sort_values(ascending=False).to_string())

    print("\nColumns: weight% is the suggested position size (inverse forecast")
    print("volatility, capped at 8%). mom_12_1% is the 12-month return excluding")
    print("the last month. vol_126% is trailing 6-month annualised volatility --")
    print("lower scores better. fcast_vol% is the model's next-month forecast.")

    outdir = DATA_DIR.parent / "reports"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / f"shortlist_{as_of.date()}.csv", index=False)
    print(f"\nwritten to artifacts/reports/shortlist_{as_of.date()}.csv")

    print("\nNot advice. This is a research screen for a paper-trading account:")
    print("a starting point for your own reading, not a buy list. The strategy")
    print("behind it did not beat a Nifty 500 index fund in backtest.")


if __name__ == "__main__":
    main()
