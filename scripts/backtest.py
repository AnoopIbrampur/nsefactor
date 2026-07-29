"""Run the factor baseline against its benchmarks and print the verdict.

Usage:  python scripts/backtest.py

Reports, in order:
  1. Information coefficient per factor -- is there any signal at all?
  2. Decile spread -- does the ranking separate winners from losers monotonically?
  3. Portfolio results vs benchmarks, gross and net of costs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nsefactor import adjust, backtest, benchmark, factors, metrics, universe
from nsefactor.config import DATA_DIR, DEFAULT_CONFIG

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("backtest")

CFG = DEFAULT_CONFIG


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def build_scores(panel: pd.DataFrame, cache: dict):
    """Score function closing over a per-date factor cache."""

    def score(p: pd.DataFrame, as_of: pd.Timestamp):
        if as_of not in cache:
            sel = universe.select(p, as_of, cfg=CFG)
            if sel.empty:
                cache[as_of] = None
                return None
            cache[as_of] = factors.compute_all(p, as_of, sel.index)
        f = cache[as_of]
        return None if f is None else f["composite"]

    return score


def main() -> None:
    panel = pd.read_parquet(DATA_DIR / "bhavcopy.parquet")
    print(f"panel: {len(panel):,} rows, {panel['date'].nunique():,} sessions")

    print("adjusting for corporate actions...")
    panel = adjust.adjusted_close(panel)
    days = pd.DatetimeIndex(sorted(panel["date"].unique()))

    rebals = backtest.month_end_dates(days)
    rebals = rebals[rebals >= days[300]]
    print(f"rebalance dates: {len(rebals)} ({rebals[0].date()} -> {rebals[-1].date()})")

    # ---- Factor panel, computed once and reused -------------------------
    print("computing factors (this takes a minute)...")
    cache: dict = {}
    rows = []
    for dt in rebals:
        sel = universe.select(panel, dt, cfg=CFG)
        if sel.empty:
            cache[dt] = None
            continue
        f = factors.compute_all(panel, dt, sel.index)
        cache[dt] = f
        rows.append(f.assign(isin=f.index))
    fpanel = pd.concat(rows, ignore_index=True)

    # ---- Forward returns for IC ----------------------------------------
    day_pos = {d: i for i, d in enumerate(days)}
    fwd = {}
    for i, dt in enumerate(rebals[:-1]):
        entry = days[day_pos[dt] + 1]
        exit_ = rebals[i + 1]
        f = cache.get(dt)
        if f is None:
            continue
        fwd[dt] = backtest.forward_returns(panel, entry, exit_, f.index)

    section("1. INFORMATION COEFFICIENT (rank correlation, factor vs next-month return)")
    print("Spearman IC per rebalance, averaged. |IC| of 0.02-0.05 is a normal,")
    print("useful equity factor. t-stat above ~2 means it is not just noise.\n")

    ic_rows = []
    for name in list(factors.FACTOR_SIGNS) + ["composite"]:
        col = f"z_{name}" if name != "composite" else "composite"
        series = []
        for dt, ret in fwd.items():
            f = cache[dt]
            if col not in f:
                continue
            joined = pd.DataFrame({"f": f[col], "r": ret}).dropna()
            if len(joined) < 30:
                continue
            series.append(joined["f"].corr(joined["r"], method="spearman"))
        s = pd.Series(series).dropna()
        if s.empty:
            continue
        tstat = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else np.nan
        ic_rows.append(
            {
                "factor": name,
                "mean_IC": round(s.mean(), 4),
                "IC_std": round(s.std(), 4),
                "t_stat": round(tstat, 2),
                "hit_rate": round((s > 0).mean(), 3),
                "n": len(s),
            }
        )
    print(pd.DataFrame(ic_rows).set_index("factor").to_string())

    section("2. DECILE SPREAD (composite score)")
    print("Average next-month return by composite decile. A working ranking is")
    print("roughly monotonic; a strong D10 with a flat middle is usually noise.\n")

    dec_rows = []
    for dt, ret in fwd.items():
        f = cache[dt]
        j = pd.DataFrame({"s": f["composite"], "r": ret}).dropna()
        if len(j) < 100:
            continue
        j["decile"] = pd.qcut(j["s"].rank(method="first"), 10, labels=range(1, 11))
        dec_rows.append(j.groupby("decile", observed=True)["r"].mean())
    dec = pd.DataFrame(dec_rows)
    tbl = pd.DataFrame(
        {
            "mean_monthly_%": (dec.mean() * 100).round(3),
            "annualised_%": ((1 + dec.mean()) ** 12 - 1).mul(100).round(2),
        }
    )
    print(tbl.to_string())
    spread = (dec[10].mean() - dec[1].mean()) * 100
    print(f"\nD10 - D1 spread: {spread:.3f}%/month ({((1+spread/100)**12-1)*100:.2f}%/yr)")

    section("3. PORTFOLIO BACKTEST")
    print(f"top {CFG.n_holdings} by composite, equal weighted, monthly rebalance,")
    print(f"entered one session after formation, {CFG.cost_bps_per_side:.0f}bp per side.\n")

    result = backtest.run(panel, build_scores(panel, cache), days, CFG)
    ledger = result["ledger"]
    if ledger.empty:
        print("no periods produced")
        return

    exits = pd.DatetimeIndex(ledger["exit"])
    rows = [
        metrics.summary(ledger["net_return"], "factor top-20 (net)"),
        metrics.summary(ledger["gross_return"], "factor top-20 (gross)"),
    ]

    eq = backtest.equal_weight_universe(panel, lambda p, d: universe.select(p, d, cfg=CFG), days, CFG)
    if not eq["ledger"].empty:
        rows.append(metrics.summary(eq["ledger"]["net_return"], "equal-weight universe (net)"))

    try:
        idx = pd.read_parquet(DATA_DIR / "indices.parquet")
        for name in ("Nifty 50", "Nifty 500"):
            try:
                s = benchmark.series(idx, name)
            except KeyError:
                continue
            price = metrics.align_monthly(s["ret"].dropna(), exits)
            tr = metrics.align_monthly(benchmark.total_return_proxy(s).dropna(), exits)
            rows.append(metrics.summary(price, f"{name} (price)"))
            rows.append(metrics.summary(tr, f"{name} (total-return approx)"))
    except FileNotFoundError:
        print("(no indices.parquet -- run the index fetch for benchmarks)\n")

    print(metrics.compare(rows).to_string())

    section("COSTS AND TURNOVER")
    print(f"mean monthly turnover : {ledger['turnover'].mean():.1%}")
    print(f"mean monthly cost     : {ledger['cost'].mean() * 100:.3f}%")
    print(f"annual cost drag      : {ledger['cost'].mean() * 12 * 100:.2f}%")
    print(f"gross CAGR            : {metrics.cagr(ledger['gross_return']) * 100:.2f}%")
    print(f"net CAGR              : {metrics.cagr(ledger['net_return']) * 100:.2f}%")

    DATA_DIR.parent.joinpath("reports").mkdir(parents=True, exist_ok=True)
    ledger.to_csv(DATA_DIR.parent / "reports" / "ledger.csv", index=False)
    print(f"\nledger written to artifacts/reports/ledger.csv")


if __name__ == "__main__":
    main()
