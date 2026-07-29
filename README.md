# nsefactor

Cross-sectional factor ranking for NSE equities, built for a swing/long-term
horizon, with an evaluation designed to be hard on itself.

The question is not "what will RELIANCE cost in March." At a monthly horizon
that forecast is close to noise, and a model that claims otherwise is usually
leaking future data. The question here is **which names in the investable
Indian universe are likely to outperform the rest over the next few months** —
a ranking problem, where errors common to the whole market cancel out.

Status: **baseline complete and evaluated.** It does not beat an index fund.
That result is reported below rather than tuned away.

## Result

Factors were selected on 2016–2020 and the table below is 2021–2026, which the
selection never saw. Top 20 equal-weighted, monthly rebalance, entered one
session after formation, 35bp per side.

| | CAGR% | Vol% | Sharpe | MaxDD% |
|---|---|---|---|---|
| mechanical selection (`vol_126`) | 5.77 | 12.72 | 0.05 | −22.72 |
| a priori (momentum + low-vol) | 12.83 | 21.99 | 0.39 | −31.60 |
| all five factors | 10.39 | 23.32 | 0.29 | −35.30 |
| equal-weight universe | 11.23 | 18.30 | 0.35 | −26.77 |
| Nifty 50 (total-return approx) | 12.18 | 13.40 | 0.49 | −14.42 |
| **Nifty 500 (total-return approx)** | **15.23** | **14.51** | **0.65** | **−17.61** |

**A Nifty 500 index fund beat every variant, on both return and risk-adjusted
return.** The best strategy variant roughly matches the Nifty 50 on CAGR while
carrying 64% more volatility and more than twice the drawdown.

The signal is not absent — it is just not enough. Composite IC over the full
sample runs t = 4.01, and the decile spread is cleanly monotonic (D1 −1.18%/yr
to D10 +17.85%/yr). What kills it is a combination of cost drag and the fact
that a monotonic decile spread among 500 names does not survive being
concentrated into 20 positions and charged 35bp a side.

### Three things this run actually taught

**A mechanical t-stat cutoff picked the worst variant.** Requiring train
t ≥ 2.0 kept only `vol_126` (t = 3.72) and discarded `mom_12_1` at t = 1.98 and
`mom_6_1` at t = 1.95 — coin-flip distinctions on 58 monthly observations. Out
of sample the kept factor decayed to t = 1.70 while the discarded momentum
factor *rose* to t = 2.31. The mechanically selected book returned 5.77%; the
one picked from published literature instead returned 12.83%.

**One factor's sign was simply wrong.** `illiq_126` was built on the Amihud
illiquidity premium and scored t = −2.67 on training data. Within a universe
already filtered to the 500 most liquid names, the illiquid tail is not a risk
premium to harvest — it is the junk end of a liquid universe. `reversal_21`
was indistinguishable from noise throughout (t = 0.39 train, 0.00 test).

**Turnover control was worth as much as signal.** The a priori and all-five
books earned almost identical gross returns (16.83% vs 16.97%) but differed by
2.4 points net, purely because one turned over 41.9%/month and the other
69.3%. Cost drag ran 1.82% to 5.82%/yr across variants.

### Caveats on the comparison

- NSE publishes only price indices here, so the benchmark total return accrues
  the published dividend yield daily. Real dividends arrive lumpily on
  ex-dates. Both bases are reported.
- The index archive is missing 1.7% of sessions in the test window.
- Strategy returns capture dividends only above the 2% corporate-action
  detection threshold, so they sit slightly below a true total return.

None of these are large enough to close a gap this size.

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

The baseline missed the bar, so the question is whether the gap is closeable or
whether the honest answer is "buy the index."

1. **Fundamentals.** The largest known limitation: with daily bars alone there
   is no size, value, or quality factor, because the bhavcopy carries no share
   count, book value, or earnings. Quality and value are the two factors most
   associated with the long-horizon investing this is built for, and they are
   entirely absent. This is the single highest-value thing to add, and it needs
   a fundamentals source.
2. **Risk-model neutralisation.** The book ran 22–23% volatility against the
   index's 14.5%, most of it an uncontrolled small-cap tilt. Sector and size
   neutralisation would test whether the ranking has any edge once that tilt is
   removed, or whether it *was* the tilt.
3. **A model, only if it earns it.** Gradient boosting on the cross-section —
   not an LSTM, since monthly cross-sections are tabular, not sequential. It
   ships only if it beats the baseline out of sample after costs, and the
   baseline currently loses to an index fund, so the bar is the index.
4. **Forward test.** Publish the monthly shortlist and paper-trade it, so the
   repo eventually reports real out-of-sample results rather than a backtest.

The bar was always a single number: does it beat a Nifty 500 index fund after
costs in a walk-forward test. It does not. Most such strategies don't, and
reporting that plainly is the point.

## Not investment advice

A research tool for a paper-trading account. Nothing here is a recommendation,
and the author is not a registered investment adviser. Distributing signals
from something like this in India requires SEBI Research Analyst registration.

## License

MIT
