"""Factor baseline, selected on a training period and reported out of sample.

Usage:  python scripts/backtest.py

The split is the point of this script. Picking which factors to keep by
looking at their full-sample information coefficient, and then reporting the
performance of that selection on the same full sample, is in-sample
selection dressed up as a result. Factors are chosen on 2016-2020 evidence
alone; everything in the final table is 2021 onward, which the selection
never saw.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nsefactor import adjust, backtest, benchmark, factors, metrics, universe
from nsefactor.config import DATA_DIR, DEFAULT_CONFIG

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

CFG = DEFAULT_CONFIG
TRAIN_END = pd.Timestamp("2020-12-31")
# Keep a factor only if its training IC is directionally right and clearly
# distinguishable from noise. 2.0 is the conventional bar.
T_STAT_BAR = 2.0
BUFFER_GRID = (1.0, 1.5, 2.0, 3.0)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def ic_table(cache: dict, fwd: dict, names, period_mask=None) -> pd.DataFrame:
    rows = []
    for name in names:
        col = f"z_{name}" if name != "composite" else "composite"
        vals = []
        for dt, ret in fwd.items():
            if period_mask is not None and not period_mask(dt):
                continue
            f = cache.get(dt)
            if f is None or col not in f:
                continue
            j = pd.DataFrame({"f": f[col], "r": ret}).dropna()
            if len(j) < 30:
                continue
            vals.append(j["f"].corr(j["r"], method="spearman"))
        s = pd.Series(vals).dropna()
        if s.empty:
            continue
        t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else np.nan
        rows.append(
            {
                "factor": name,
                "mean_IC": round(s.mean(), 4),
                "t_stat": round(t, 2),
                "hit_rate": round((s > 0).mean(), 3),
                "n": len(s),
            }
        )
    return pd.DataFrame(rows).set_index("factor")


def main() -> None:
    panel = pd.read_parquet(DATA_DIR / "bhavcopy.parquet")
    # Bridge ISIN changes before adjusting: a company that switched ISIN
    # otherwise carries a truncated history and looks newly listed.
    panel = adjust.apply_isin_links(panel)
    panel = adjust.adjusted_close(panel)
    days = pd.DatetimeIndex(sorted(panel["date"].unique()))

    rebals = backtest.month_end_dates(days)
    rebals = rebals[rebals >= days[300]]
    print(f"panel {len(panel):,} rows | {len(rebals)} rebalances "
          f"{rebals[0].date()} -> {rebals[-1].date()}")
    print(f"train <= {TRAIN_END.date()} | test > {TRAIN_END.date()}")

    print("\ncomputing factors...")
    cache: dict = {}
    for dt in rebals:
        sel = universe.select(panel, dt, cfg=CFG)
        cache[dt] = None if sel.empty else factors.compute_all(panel, dt, sel.index)

    day_pos = {d: i for i, d in enumerate(days)}
    fwd = {}
    for i, dt in enumerate(rebals[:-1]):
        f = cache.get(dt)
        if f is None:
            continue
        fwd[dt] = backtest.forward_returns(panel, days[day_pos[dt] + 1], rebals[i + 1], f.index)

    all_names = list(factors.FACTOR_SIGNS)

    # ---- 1. Selection, on training data only ---------------------------
    section("1. FACTOR SELECTION (training period only: 2016-2020)")
    train_ic = ic_table(cache, fwd, all_names, lambda d: d <= TRAIN_END)
    print(train_ic.to_string())

    keep = tuple(train_ic.index[train_ic["t_stat"] >= T_STAT_BAR])
    dropped = [n for n in all_names if n not in keep]
    print(f"\nkeeping (train t >= {T_STAT_BAR}): {list(keep)}")
    for n in dropped:
        t = train_ic.loc[n, "t_stat"]
        why = "wrong sign" if t <= -T_STAT_BAR else "indistinguishable from noise"
        print(f"dropping {n:12} t={t:6.2f}  ({why})")

    if not keep:
        print("\nno factor cleared the bar on training data. Stopping.")
        return

    # ---- 2. Does the selection hold out of sample? ----------------------
    section("2. SAME FACTORS, TEST PERIOD (2021 onward) -- not used for selection")
    test_ic = ic_table(cache, fwd, all_names, lambda d: d > TRAIN_END)
    print(test_ic.to_string())
    print("\nFactors kept above should still show t >= 2 here. Ones that do not")
    print("were training-period artefacts, and that is worth knowing.")

    # Rebuild composites from the kept factors only.
    for dt, f in cache.items():
        if f is None:
            continue
        zc = [f"z_{n}" for n in keep]
        enough = f[zc].notna().sum(axis=1) >= len(zc) / 2
        f["composite"] = f[zc].mean(axis=1, skipna=True).where(enough)

    section("3. COMPOSITE IC, TRAIN vs TEST")
    comp = pd.concat(
        [
            ic_table(cache, fwd, ["composite"], lambda d: d <= TRAIN_END).assign(period="train"),
            ic_table(cache, fwd, ["composite"], lambda d: d > TRAIN_END).assign(period="test"),
        ]
    )
    print(comp.set_index("period", append=True).to_string())

    # ---- 3. Turnover buffer, tuned on train -----------------------------
    def score_fn(p, as_of):
        f = cache.get(as_of)
        return None if f is None else f["composite"]

    section("4. TURNOVER BUFFER (chosen on training period only)")
    print("Unbuffered, the book turns over ~69%/month and costs ~5.8%/yr. A")
    print("buffer holds a name until it leaves the top N x mult, so churn at")
    print("the rank boundary stops being paid for.\n")

    grid = []
    for mult in BUFFER_GRID:
        r = backtest.run(panel, score_fn, days, CFG, end=TRAIN_END, buffer_mult=mult)
        led = r["ledger"]
        if led.empty:
            continue
        grid.append(
            {
                "buffer": mult,
                "turnover%": round(led["turnover"].mean() * 100, 1),
                "cost_drag%": round(led["cost"].mean() * 12 * 100, 2),
                "gross_CAGR%": round(metrics.cagr(led["gross_return"]) * 100, 2),
                "net_CAGR%": round(metrics.cagr(led["net_return"]) * 100, 2),
                "net_Sharpe": round(metrics.sharpe(led["net_return"]), 2),
            }
        )
    gdf = pd.DataFrame(grid).set_index("buffer")
    print(gdf.to_string())
    best_buffer = float(gdf["net_Sharpe"].idxmax())
    print(f"\nbest buffer on train by net Sharpe: {best_buffer}")

    # ---- 4. The actual result -------------------------------------------
    section(f"5. OUT-OF-SAMPLE RESULT (2021+, buffer {best_buffer} fixed on train)")
    print("Three factor sets, all evaluated on the same untouched test period.")
    print("A single t-stat cutoff is a coin flip for factors near the bar, so")
    print("the sensitivity to that choice is reported rather than hidden:\n")
    print("  mechanical  -- whatever cleared train t >= 2.0")
    print("  a priori    -- momentum + low-vol, chosen from the published")
    print("                 literature rather than from this sample at all")
    print("  all five    -- no selection\n")

    variants = {
        f"mechanical {list(keep)}": keep,
        "a priori (mom_12_1 + vol_126)": ("mom_12_1", "vol_126"),
        "all five factors": tuple(all_names),
    }

    rows = []
    ledgers = {}
    for label, subset in variants.items():
        for dt, f in cache.items():
            if f is None:
                continue
            zc = [f"z_{n}" for n in subset]
            enough = f[zc].notna().sum(axis=1) >= len(zc) / 2
            f["composite"] = f[zc].mean(axis=1, skipna=True).where(enough)

        r = backtest.run(panel, score_fn, days, CFG, start=TRAIN_END, buffer_mult=best_buffer)
        if r["ledger"].empty:
            continue
        ledgers[label] = r["ledger"]
        rows.append(metrics.summary(r["ledger"]["net_return"], f"{label} (net)"))

    led = list(ledgers.values())[0]
    exits = pd.DatetimeIndex(led["exit"])

    eq = backtest.equal_weight_universe(
        panel, lambda p, d: universe.select(p, d, cfg=CFG), days, CFG, start=TRAIN_END
    )
    if not eq["ledger"].empty:
        rows.append(metrics.summary(eq["ledger"]["net_return"], "equal-weight universe (net)"))

    try:
        idx = pd.read_parquet(DATA_DIR / "indices.parquet")
        for name in ("Nifty 50", "Nifty 500"):
            try:
                s = benchmark.series(idx, name)
            except KeyError:
                continue
            rows.append(metrics.summary(metrics.align_monthly(s["ret"].dropna(), exits), f"{name} (price)"))
            rows.append(
                metrics.summary(
                    metrics.align_monthly(benchmark.total_return_proxy(s).dropna(), exits),
                    f"{name} (total-return approx)",
                )
            )
    except FileNotFoundError:
        print("(indices.parquet missing -- no index benchmarks)\n")

    print(metrics.compare(rows).to_string())

    section("COSTS AND TURNOVER, BY VARIANT")
    for label, l in ledgers.items():
        print(
            f"{label:34} turnover {l['turnover'].mean():5.1%}/mo  "
            f"drag {l['cost'].mean() * 12 * 100:4.2f}%/yr  "
            f"gross {metrics.cagr(l['gross_return']) * 100:5.2f}%  "
            f"net {metrics.cagr(l['net_return']) * 100:5.2f}%"
        )

    out = DATA_DIR.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    led.to_csv(out / "ledger_oos.csv", index=False)
    print(f"\nledger -> artifacts/reports/ledger_oos.csv")


if __name__ == "__main__":
    main()
