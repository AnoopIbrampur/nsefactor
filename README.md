# nsefactor

Cross-sectional factor ranking for NSE equities, built for a swing/long-term
horizon, with an evaluation designed to be hard on itself.

The question is not "what will RELIANCE cost in March." At a monthly horizon
that forecast is close to noise, and a model that claims otherwise is usually
leaking future data. The question here is **which names in the investable
Indian universe are likely to outperform the rest over the next few months** —
a ranking problem, where errors common to the whole market cancel out.

Status: **data layer complete and tested.** Factors and backtest are next.

## The panel

2,865 NSE trading sessions, 2015-01-01 to 2026-07-27, rebuilt from daily
bhavcopy archives. 4.47M rows, 3,256 ISINs, zero nulls. Yearly coverage gaps
run 9–16 days, which is exactly NSE's trading-holiday count — no silent holes.

| Check | Result |
|---|---|
| Corporate actions detected | 4,933 across 1,399 ISINs (430/yr) |
| Daily returns below −40%, raw | 1,052 |
| Daily returns below −40%, adjusted | **293** |
| Return std, raw → adjusted | 0.0482 → **0.0319** |
| Universe size | 481 min / 500 median |
| Monthly universe churn | 3.4% mean, 5.2% max |
| Overlap with published Nifty 500 | 80.6% |

The two numbers that matter most:

**933.** ISINs that were in the investable universe at some point but are not
today (1,433 ever vs 500 in the final month). A backtest built on today's
constituent list silently deletes all 933.

**759 of 1,052.** Phantom crashes removed by corporate-action adjustment. Each
one was a split or bonus that an unadjusted momentum factor would have read as
a catastrophic loss on a day nothing happened.

## Why the data layer got built first

Most Indian-equity backtests on GitHub are wrong before any model is trained,
in three specific ways. This repo addresses each one in code rather than in a
disclaimer.

### 1. Survivorship bias

The usual approach downloads today's Nifty 500 constituent list and applies it
to a ten-year backtest. Every company that was delisted, acquired, or wiped out
has already been removed from that list, so the backtest earns returns nobody
holding the portfolio could have earned.

We never use the published list for selection. A bhavcopy dated 2016-03-31
contains exactly the stocks that traded on 2016-03-31, *including* the ones
that no longer exist. Ranking that file by liquidity reconstructs the universe
as it actually stood. No forward-looking information, and no historical
index-membership feed to source.

`tests/test_data.py::TestUniverseCausality` pins this down: selecting from a
panel truncated at the as-of date must return exactly what selecting from the
full panel returns, and destroying a stock *after* the as-of date must not
change what was selected before it.

### 2. Corporate actions

NSE publishes no adjusted price series, and its corporate-actions feed sits
behind the cookie-gated `www` host. Neither is needed. On the ex-date of a
split, bonus, or dividend, NSE reports `prevclose` as the *adjusted* prior
close while the previous session's `close` is the raw figure:

```
factor = prevclose(t) / close(t-1)
```

A 1:2 split reads as 0.5, a 1:10 as 0.1, a 4% dividend as 0.96. Chaining these
backwards yields an adjusted close built entirely from data already on disk.
Across the clean sample this ratio is *exactly* 1.0 for 99.5% of rows, which is
what makes the remaining 0.5% trustworthy as real actions.

This matters because on an unadjusted series a 1:5 split looks like a −80%
day, and any momentum factor will rank that stock as the worst in the market on
precisely the day nothing bad happened.

### 3. Symbol reuse

NSE recycles ticker symbols after delistings — 384 symbols in this panel map to
more than one ISIN. The entire panel is therefore keyed on ISIN, never symbol.
Symbols are carried along for display only.

## The NSE trading calendar is not "weekdays minus holidays"

NSE runs a ceremonial one-hour *Muhurat* session on Diwali, and it usually
falls on a **Saturday or Sunday**. It is a real settled session with its own
bhavcopy.

An early version of the fetcher iterated weekdays only. The missing Muhurat
sessions produced a hole in the panel, so the next session's `prevclose`
referred to a day that wasn't there — surfacing as a cluster of phantom
corporate actions on the first weekday after Diwali every year (2016-11-01,
2019-10-29, 2020-11-17). The fetcher now requests every calendar day and lets
NSE's 404 define the calendar. Regression test:
`TestCalendarCoverage::test_muhurat_weekend_session_is_requested`.

A related rule: a 404 means "no session that day," a 403 means "we are being
throttled." Conflating them punches silent holes in the price history, so 403s
retry with backoff and then raise rather than being recorded as holidays.

## Data source

Everything comes from `nsearchives.nseindia.com`, which serves daily bhavcopy
archives without the cookie handshake that `www.nseindia.com` requires. Two
incompatible layouts are normalised to one schema:

| Era | Format | Path |
|---|---|---|
| through 2023 | legacy | `/content/historical/EQUITIES/{YYYY}/{MON}/cm{DDMMMYYYY}bhav.csv.zip` |
| 2024 onward | UDiFF | `/content/cm/BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip` |

Both carry ISIN, so they merge without an external symbol map. Filtering keeps
the `EQ` rolling-settlement series and ISINs beginning `INE` — ETFs and fund
units (`INF...`) share the EQ series but are not company equity.

The archive host throttles above a few concurrent requests and will IP-ban for
a while if pushed. `max_workers` defaults to 4; raw zips are cached under
`artifacts/raw/`, so re-runs are nearly free and `load_cached()` rebuilds the
panel with no network at all.

## Usage

```bash
pip install -e ".[dev]"
python scripts/fetch.py 2015-01-01     # ~2,900 sessions, cached to disk
python scripts/validate.py             # data-quality report
pytest -q
```

## Layout

```
src/nsefactor/
  config.py     paths, universe filters, cost assumptions
  bhavcopy.py   download + normalise both archive formats
  adjust.py     corporate-action factors recovered from prevclose
  universe.py   point-in-time liquidity-ranked universe
  cli.py        nsefactor fetch | validate | universe
scripts/
  fetch.py      build the panel
  validate.py   coverage, corporate actions, universe churn
tests/
  test_data.py  causality, adjustment, calendar invariants
```

## What comes next

1. **Factor baseline, no ML.** 12-1 momentum, quality, low volatility, ranked
   cross-sectionally, monthly rebalance, costs charged at 35bp per side
   (brokerage + exchange fees + STT + impact). Benchmarked against Nifty 50 and
   equal-weight.
2. **A model, only if it earns it.** Gradient boosting on the cross-section —
   not an LSTM; monthly cross-sections are a tabular problem, not a sequence
   one. It ships only if it beats step 1 out-of-sample after costs.
3. **Forward test.** Publish the monthly shortlist and paper-trade it, so the
   repo eventually reports real out-of-sample results rather than a backtest.

The bar for step 1 is a single number: does it beat a Nifty 50 index fund after
costs in a walk-forward test. Most such strategies don't, and reporting that
honestly is the point.

## Not investment advice

A research tool for a paper-trading account. Nothing here is a recommendation,
and the author is not a registered investment adviser. Distributing signals
from something like this in India requires SEBI Research Analyst registration.

## License

MIT
