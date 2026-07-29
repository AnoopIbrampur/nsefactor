"""Data-quality report for the bhavcopy panel.

Run before trusting any backtest built on this data. Everything here is a
check that would otherwise fail silently and flatter the results.
"""

from __future__ import annotations

import logging

import pandas as pd

from nsefactor import adjust, bhavcopy as bc, universe
from nsefactor.config import DATA_DIR, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("validate")

# NSE observes roughly this many trading holidays a year on weekdays.
EXPECTED_HOLIDAYS_PER_YEAR = (10, 20)


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def main() -> None:
    panel = pd.read_parquet(DATA_DIR / "bhavcopy.parquet")
    days = bc.trading_days(panel)

    section("COVERAGE")
    weekdays = pd.Series(
        1, index=[d for d in pd.date_range(days[0], days[-1]) if d.weekday() < 5]
    )
    have = pd.Series(1, index=days).resample("YE").sum()
    want = weekdays.resample("YE").sum()
    cov = pd.DataFrame({"trading_days": have, "weekdays": want}).fillna(0).astype(int)
    cov["gap"] = cov["weekdays"] - cov["trading_days"]
    cov.index = cov.index.year
    print(cov.to_string())

    lo, hi = EXPECTED_HOLIDAYS_PER_YEAR
    full_years = cov.iloc[1:-1] if len(cov) > 2 else cov
    suspicious = full_years[(full_years["gap"] < lo) | (full_years["gap"] > hi)]
    if len(suspicious):
        print(f"\n!! {len(suspicious)} year(s) with gaps outside the {lo}-{hi} holiday range:")
        print(suspicious.to_string())
    else:
        print(f"\nOK: every full year's gap sits within {lo}-{hi} trading holidays.")

    section("PANEL SHAPE")
    print(f"rows              {len(panel):,}")
    print(f"date range        {days[0].date()} -> {days[-1].date()}")
    print(f"trading days      {len(days):,}")
    print(f"unique ISINs      {panel['isin'].nunique():,}")
    print(f"unique symbols    {panel['symbol'].nunique():,}")
    print(f"nulls             {int(panel.isna().sum().sum()):,}")
    listed = panel.groupby("date")["isin"].nunique()
    print(f"names per day     {listed.min()} min / {int(listed.median())} median / {listed.max()} max")

    section("SYMBOL REUSE (why the panel is keyed on ISIN)")
    per_symbol = panel.groupby("symbol")["isin"].nunique()
    reused = per_symbol[per_symbol > 1]
    print(f"symbols mapping to >1 ISIN: {len(reused)}")
    if len(reused):
        print(reused.sort_values(ascending=False).head(10).to_string())

    section("CORPORATE ACTIONS")
    actions = adjust.detect_factors(panel)
    print(f"detected          {len(actions):,} across {actions['isin'].nunique():,} ISINs")
    print(f"per year          {len(actions) / (len(days) / 250):.0f}")
    print("\nfactor distribution:")
    print(actions["factor"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
    print("\nlargest (splits/bonuses):")
    print(actions.nsmallest(8, "factor").to_string(index=False))

    section("ADJUSTED RETURNS")
    adj = adjust.adjusted_returns(adjust.adjusted_close(panel))
    raw_ret = panel.sort_values(["isin", "date"]).groupby("isin")["close"].pct_change()
    for label, series in (("raw", raw_ret), ("adjusted", adj["ret"])):
        s = series.dropna()
        extreme = (s < -0.4).sum()
        print(f"{label:9} n={len(s):>9,}  std={s.std():.4f}  days below -40%: {extreme:,}")
    print("\nA large drop in the '-40%' count is the corporate-action fix working.")

    section("POINT-IN-TIME UNIVERSE")
    rebal = pd.DatetimeIndex(
        pd.Series(days).groupby(pd.Series(days).dt.to_period("M")).last()
    )
    hist = universe.build_history(panel, rebal)
    sizes = hist.groupby("date").size()
    print(f"rebalance dates   {len(rebal)}")
    print(f"universe size     {sizes.min()} min / {int(sizes.median())} median / {sizes.max()} max")

    churn = universe.turnover_rate(hist)
    print(f"monthly churn     mean {churn.mean():.1%}, max {churn.max():.1%}")

    ever = hist["isin"].nunique()
    latest = set(hist.loc[hist["date"] == hist["date"].max(), "isin"])
    print(f"ISINs ever in universe: {ever:,} (vs {len(latest)} in the final month)")
    print(f"-> {ever - len(latest):,} names were investable at some point but are not today.")
    print("   Those are precisely what a current-constituents backtest deletes.")

    section("UNIVERSE vs PUBLISHED NIFTY 500 (sanity check only)")
    try:
        idx = universe.fetch_current_index()
        latest_sel = universe.select(panel, days[-1])
        overlap = universe.overlap_with_index(latest_sel, idx)
        print(f"our top-{DEFAULT_CONFIG.universe_size} vs published Nifty 500: {overlap:.1%} overlap")
        print("Disagreement is expected -- the index applies float, sector, and")
        print("listing-history rules we deliberately do not model.")
    except Exception as exc:
        print(f"could not fetch index list ({exc}); skipping")


if __name__ == "__main__":
    main()
