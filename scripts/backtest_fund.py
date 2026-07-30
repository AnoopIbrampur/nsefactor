"""Evaluate value and quality factors alongside the price factors.

Usage:  python scripts/backtest_fund.py

On the strength of the evidence
-------------------------------
The 2021+ test period has already been examined three times while evaluating
price-only factors, so any *further* claim about those factors from this data is
weak. The fundamental factors are different: they have never been computed on
this sample at all, let alone evaluated on it. Their first out-of-sample
reading is genuinely fresh evidence.

To keep that distinction intact this script does not tune anything on the test
period. Factor selection is made on 2016-2020 evidence and on published priors,
the construction settings are inherited unchanged from the price-only work, and
the test period is read exactly once per configuration.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nsefactor import (
    adjust,
    backtest,
    benchmark,
    factors,
    fundamentals as F,
    fundfactors as FF,
    metrics,
    universe,
)
from nsefactor.config import DATA_DIR, DEFAULT_CONFIG

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("backtest_fund")

CFG = DEFAULT_CONFIG
TRAIN_END = pd.Timestamp("2020-12-31")

PRICE_FACTORS = ("mom_12_1", "vol_126")
VALUE_FACTORS = ("earnings_yield", "book_to_price")
QUALITY_FACTORS = ("roe", "gross_margin", "debt_to_equity", "earnings_stability")


def section(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def ic_table(cache: dict, fwd: dict, names, mask=None) -> pd.DataFrame:
    rows = []
    for name in names:
        col = f"z_{name}" if name != "composite" else "composite"
        vals, coverage = [], []
        for dt, ret in fwd.items():
            if mask is not None and not mask(dt):
                continue
            f = cache.get(dt)
            if f is None or col not in f:
                continue
            j = pd.DataFrame({"f": f[col], "r": ret}).dropna()
            if len(j) < 30:
                continue
            vals.append(j["f"].corr(j["r"], method="spearman"))
            coverage.append(len(j) / max(1, len(f)))
        s = pd.Series(vals).dropna()
        if s.empty:
            continue
        t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else np.nan
        rows.append(
            {
                "factor": name,
                "mean_IC": round(s.mean(), 4),
                "t_stat": round(t, 2),
                "hit": round((s > 0).mean(), 3),
                "coverage": round(float(np.mean(coverage)), 3),
                "n": len(s),
            }
        )
    return pd.DataFrame(rows).set_index("factor")


def main() -> None:
    fund_path = F.FUND_DIR / "fundamentals.parquet"
    if not fund_path.exists():
        raise SystemExit("run scripts/fetch_fundamentals.py first")

    panel = pd.read_parquet(DATA_DIR / "bhavcopy.parquet")
    panel = adjust.apply_isin_links(panel)
    panel = adjust.adjusted_close(panel)
    fund = pd.read_parquet(fund_path)

    days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rebals = backtest.month_end_dates(days)
    rebals = rebals[rebals >= days[300]]
    # Fundamentals coverage ends when NSE migrated its filing system, so the
    # evaluation stops there rather than silently ranking on stale accounts.
    fund_end = fund["broadcast_date"].max()
    rebals = rebals[rebals <= fund_end]

    print(f"panel {len(panel):,} rows | fundamentals {len(fund):,} filings, "
          f"{fund['isin'].nunique():,} ISINs")
    print(f"rebalances {len(rebals)}  {rebals[0].date()} -> {rebals[-1].date()}")
    print(f"train <= {TRAIN_END.date()} | test > {TRAIN_END.date()}")

    all_names = list(factors.FACTOR_SIGNS) + list(FF.FUND_FACTOR_SIGNS)

    print("\ncomputing price and fundamental factors...")
    cache: dict = {}
    for dt in rebals:
        sel = universe.select(panel, dt, cfg=CFG)
        if sel.empty:
            cache[dt] = None
            continue
        fu = FF.compute(panel, fund, dt, sel.index)
        cache[dt] = factors.compute_all(
            panel,
            dt,
            sel.index,
            extra=fu[list(FF.FUND_FACTOR_SIGNS)] if not fu.empty else None,
            extra_signs=FF.FUND_FACTOR_SIGNS,
            require_all=False,
        )

    day_pos = {d: i for i, d in enumerate(days)}
    fwd = {}
    for i, dt in enumerate(rebals[:-1]):
        f = cache.get(dt)
        if f is None:
            continue
        fwd[dt] = backtest.forward_returns(panel, days[day_pos[dt] + 1], rebals[i + 1], f.index)

    # ---- Coverage: how many names actually have usable fundamentals -----
    section("1. FUNDAMENTAL COVERAGE")
    print("A factor is only useful if it exists for enough of the universe.")
    print("Coverage below is the share of the investable universe with a value.\n")
    cov_rows = []
    for name in FF.FUND_FACTOR_SIGNS:
        vals = [f[name].notna().mean() for f in cache.values() if f is not None and name in f]
        if vals:
            cov_rows.append({"factor": name, "mean_coverage": round(float(np.mean(vals)), 3),
                             "min": round(float(np.min(vals)), 3),
                             "max": round(float(np.max(vals)), 3)})
    print(pd.DataFrame(cov_rows).set_index("factor").to_string())

    # ---- IC, train then test --------------------------------------------
    section("2. INFORMATION COEFFICIENT, TRAINING PERIOD (2016-2020)")
    train_ic = ic_table(cache, fwd, all_names, lambda d: d <= TRAIN_END)
    print(train_ic.to_string())

    section("3. INFORMATION COEFFICIENT, TEST PERIOD (2021+)")
    print("First reading for every fundamental factor. The price factors have")
    print("been examined here before, so treat only the new rows as fresh.\n")
    test_ic = ic_table(cache, fwd, all_names, lambda d: d > TRAIN_END)
    print(test_ic.to_string())

    # ---- Composites, chosen a priori ------------------------------------
    section("4. COMPOSITE PORTFOLIOS (2021+)")
    print("Factor groups fixed in advance from the literature, not screened here.")
    print("Construction settings inherited from the price-only work: 20 names,")
    print("equal weighted, monthly, entered t+1, 35bp/side.\n")

    combos = {
        "price only (mom + low-vol)": PRICE_FACTORS,
        "value only": VALUE_FACTORS,
        "quality only": QUALITY_FACTORS,
        "value + quality": VALUE_FACTORS + QUALITY_FACTORS,
        "price + value + quality": PRICE_FACTORS + VALUE_FACTORS + QUALITY_FACTORS,
    }

    def score_fn(p, as_of):
        f = cache.get(as_of)
        return None if f is None else f["composite"]

    rows, ledgers = [], {}
    for label, subset in combos.items():
        for dt, f in cache.items():
            if f is None:
                continue
            zc = [f"z_{n}" for n in subset if f"z_{n}" in f.columns]
            if not zc:
                f["composite"] = np.nan
                continue
            # Require at least half of the chosen factors, otherwise a name with
            # one stray value competes against names scored on everything.
            enough = f[zc].notna().sum(axis=1) >= max(1, len(zc) / 2)
            f["composite"] = f[zc].mean(axis=1, skipna=True).where(enough)

        r = backtest.run(panel, score_fn, days, CFG, start=TRAIN_END, buffer_mult=1.0)
        led = r["ledger"]
        if led.empty:
            continue
        ledgers[label] = led
        rows.append(metrics.summary(led["net_return"], f"{label} (net)"))

    if not ledgers:
        print("no portfolios produced")
        return

    exits = pd.DatetimeIndex(list(ledgers.values())[0]["exit"])
    eqw = backtest.equal_weight_universe(
        panel, lambda p, d: universe.select(p, d, cfg=CFG), days, CFG, start=TRAIN_END
    )
    if not eqw["ledger"].empty:
        rows.append(metrics.summary(eqw["ledger"]["net_return"], "equal-weight universe (net)"))

    try:
        idx = pd.read_parquet(DATA_DIR / "indices.parquet")
        for nm in ("Nifty 50", "Nifty 500"):
            try:
                s = benchmark.series(idx, nm)
            except KeyError:
                continue
            rows.append(metrics.summary(
                metrics.align_monthly(benchmark.total_return_proxy(s).dropna(), exits),
                f"{nm} (total-return approx)"))
    except FileNotFoundError:
        print("(no indices.parquet -- index benchmarks unavailable)\n")

    print(metrics.compare(rows).to_string())

    section("TURNOVER AND COSTS")
    for label, led in ledgers.items():
        print(f"{label:32} turnover {led['turnover'].mean():5.1%}/mo  "
              f"drag {led['cost'].mean() * 12 * 100:4.2f}%/yr  "
              f"gross {metrics.cagr(led['gross_return']) * 100:6.2f}%  "
              f"net {metrics.cagr(led['net_return']) * 100:6.2f}%")

    out = DATA_DIR.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    train_ic.to_csv(out / "fund_ic_train.csv")
    test_ic.to_csv(out / "fund_ic_test.csv")
    print(f"\nIC tables -> artifacts/reports/fund_ic_{{train,test}}.csv")


if __name__ == "__main__":
    main()
