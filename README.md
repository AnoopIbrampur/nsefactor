# nsefactor

Two forecasting questions on 11 years of NSE equity data, and two different
answers.

**Which stocks will outperform?** Attempted first, as a cross-sectional factor
ranking. The answer, out of sample, is that a Nifty 500 index fund beats it.
Reported in full below rather than tuned away.

**How volatile will each stock be next month?** This one works. Gradient
boosting on a log-volatility-ratio target beats RiskMetrics EWMA by 11.5% on
RMSE, naive persistence by 18.4%, and GARCH(1,1) by 27.3% — significant when
clustered by date (16/19 months, p = 0.0003). But the edge is entirely mean
reversion: it predicts elevated volatility subsiding, not volatility spiking.

The through-line is the useful part: at these horizons *returns* are close to
noise while *risk* is forecastable. The same thing was true of hourly crypto,
where this repo's author found the price task a three-way tie with persistence
and the volatility task an 8–14% win.

> **Numbers below predate the ISIN-bridging fix.** 368 companies had their
> price history split in two by an ISIN change (usually a face-value split), so
> momentum was null or truncated for those names. The tables in both result
> sections were produced before that was found and are not yet regenerated.
>
> This matters, and not hypothetically: on the buggy data, risk-weighted
> 60-name construction appeared to beat the equal-weighted 20-name book on
> Sharpe (0.47 vs 0.39). After the fix it is the reverse — 0.28 vs 0.43. The
> headline conclusions are unchanged (volatility is forecastable, the factor
> ranking loses to a Nifty 500 index fund), but the individual figures will
> move. Regenerate with `python scripts/vol_model.py` and
> `python scripts/backtest.py`.

---

## Result 1: volatility forecasting (works)

21-day-ahead realised volatility, chronological 70/15/15 split, evaluated on
**non-overlapping** forward windows in 2024-12 → 2026-07.

| | RMSE | MAE | corr | bias |
|---|---|---|---|---|
| persistence (trailing 21d vol) | 0.16162 | 0.11209 | 0.448 | +0.003 |
| EWMA (RiskMetrics, λ=0.94) | 0.14898 | 0.10347 | 0.486 | +0.008 |
| **gradient boosting** | **0.13184** | **0.09211** | **0.528** | −0.021 |

**+18.4% vs persistence, +11.5% vs EWMA.**

Significance is computed per date, not per row. 9,531 stock-months is not 9,531
independent observations — stocks move together, so the effective sample is
closer to the 19 evaluation dates:

- beats EWMA on **16 of 19** months
- sign test p = 0.0022, paired t-test t = −4.45, p = **0.00031**
- mean per-date improvement 13.3%, worst month −13.9%, best +30.6%

### Against GARCH(1,1)

GARCH needs a refit per stock per date, so it runs on a 25-stock subsample. It
converged on 192 of 292 rows; **all methods are scored on those 192 rows only**,
since comparing GARCH on its own successful fits against other methods on every
row would hand it a self-selected easier sample.

| | RMSE | MAE | corr | bias |
|---|---|---|---|---|
| persistence | 0.14504 | 0.10434 | 0.403 | −0.001 |
| EWMA | 0.13806 | 0.09871 | 0.406 | +0.009 |
| GARCH(1,1) | 0.16659 | 0.12886 | 0.295 | **+0.082** |
| **gradient boosting** | **0.12105** | **0.08905** | **0.467** | −0.014 |

GARCH lands worse than naive persistence here, and the +0.082 bias says why: a
21-day-ahead GARCH forecast reverts toward the unconditional variance, which
sits well above realised volatility for most names in this universe. Worth
reporting rather than quietly dropping the baseline that lost.

### The honest limitation

The edge is mean reversion, and it is concentrated where it is least needed:

| | months | mean improvement |
|---|---|---|
| volatility **fell** month-over-month | 13 | **+18.3%** |
| volatility **rose** month-over-month | 6 | **+2.3%** |

Correlation between month-over-month vol change and model improvement is
**−0.63 (p = 0.004)**. The model is good at knowing when elevated volatility
will subside and adds almost nothing when volatility jumps — which is exactly
when a risk forecast matters most.

So this is a **position-sizing tool, not a crash warning system.** Worth being
blunt about, because the failure mode of a risk model that quietly stops
working during turbulence is much worse than one that never worked.

### Two design choices that carried the result

**Predict a ratio, not a level.** The target is
`log(forward_vol / ewma_anchor)`. A single model trained across 1,352 stocks
whose baseline volatilities span an order of magnitude cannot share an
intercept — it learns a compromise level that biases every name. Predicting
change relative to a per-stock anchor removes that entirely. This is the same
bug, and the same fix, as the per-coin scaling bias in the crypto version.

**Anchor on EWMA, not trailing realised vol.** Trailing realised vol is noisy,
and a noisy denominator amplifies error straight into the target. EWMA is
smoother and makes the better anchor.

Permutation importance, measured on the test slice the model never trained on,
puts `ewma` far ahead of everything else (+1.00), then `parkinson_21` (+0.44)
and `rv_ratio_21_126` (+0.04). The anchor mattering most *is* the mean-reversion
effect showing up directly: the level tells the model how much reversion to
expect. `parkinson_21` earning second place is the intraday high-low range
carrying information the close-to-close series throws away.

Two features score slightly negative and are doing nothing: `market_rv_21`
(−0.014) and `rv_21` (−0.0002). The cross-sectional market regime is already
implied by each stock's own EWMA, so it adds no independent signal here.

---

## Result 2: fundamentals (do not close the gap)

The price-only ranking lost to an index fund, and the obvious explanation was
the missing half of the toolkit: with daily bars alone there is no value or
quality factor, and those are the two families most associated with
long-horizon investing. So they were built properly, from NSE's XBRL filings,
with every figure stamped by the date it was actually broadcast.

They did not help.

Test period 2022-01 to 2025-02, 39 months, factors and construction fixed in
advance:

| | CAGR% | Vol% | Sharpe | MaxDD% |
|---|---|---|---|---|
| price only (momentum + low-vol) | 8.45 | 24.82 | 0.22 | −34.04 |
| fundamentals only | 7.40 | 19.77 | 0.16 | −23.52 |
| price + fundamentals | 3.73 | 18.38 | −0.03 | −24.64 |
| equal-weight universe | 6.13 | 18.93 | 0.10 | −27.64 |
| Nifty 50 (total-return approx) | 11.49 | 13.37 | 0.44 | −13.82 |
| **Nifty 500 (total-return approx)** | **13.28** | 14.89 | **0.52** | −17.61 |

Combining price and fundamental factors was *worse than either alone*.

No fundamental factor cleared t = 2 on a meaningful sample. Earnings yield came
closest at t = 1.88 over 37 months, having scored a training IC of exactly zero.
`asset_growth` shows t = 2.95 with a perfect hit rate on **four observations**,
which is noise, not a result — it is in the table only because hiding it would
be worse.

The one thing fundamentals did deliver is stability: turnover fell from 42.6% to
12.8% a month and cost drag from 3.58% to 1.08% a year, with lower volatility
and a shallower drawdown. Steadier signals, not more profitable ones.

### What limits this conclusion

Three constraints, and they are properties of the data rather than fixable:

* **XBRL does not exist before 2018** — 0% coverage 2015–2017, 54.5% in 2018,
  ~99% from 2020. The study is six years, not eleven.
* **Balance-sheet items only entered the taxonomy around 2022**, so
  book-to-price, ROE, leverage and asset growth have roughly three years of
  history. That cannot be split into train and test honestly, so they are
  reported descriptively and never traded here. The two most interesting value
  and quality factors are effectively untestable on this source.
* **Coverage tops out near 57%** of the investable universe even for the
  P&L-only factors, because just 813 of 1,236 universe ISINs have XBRL filings
  at all.

And the window matters: 2022–2025 in India was a strong momentum and small-cap
market, a regime in which value and quality have historically lagged. This is
one regime's evidence, not a verdict on the factors in general.

---

## Result 3: factor ranking (does not work)

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
pip install -e ".[dev,garch]"
python scripts/fetch.py 2015-01-01     # ~2,900 sessions, cached to disk
python scripts/validate.py             # data-quality report
python scripts/vol_model.py            # volatility forecast vs baselines
python scripts/backtest.py             # factor ranking vs index
pytest -q                              # 62 tests
```

## Layout

```
src/nsefactor/
  config.py       paths, universe filters, cost assumptions
  bhavcopy.py     download + normalise both archive formats
  adjust.py       corporate-action factors recovered from prevclose
  universe.py     point-in-time liquidity-ranked universe
  benchmark.py    NSE index levels + total-return approximation
  volatility.py   vol estimators, features, log-ratio target, baselines
  factors.py      cross-sectional price factors
  backtest.py     walk-forward engine with costs and rank buffering
  metrics.py      CAGR, Sharpe, drawdown, monthly alignment
  cli.py          nsefactor fetch | validate | universe
scripts/
  fetch.py        build the panel
  validate.py     coverage, corporate actions, universe churn
  vol_model.py    volatility forecast vs persistence / EWMA / GARCH
  backtest.py     factor ranking vs index, train/test split
tests/
  test_data.py        causality, adjustment, calendar invariants
  test_backtest.py    factor causality, timing, cost accounting
  test_volatility.py  forward-target isolation, feature causality
```

## What comes next

The volatility model works and the factor model does not, so the roadmap
follows the volatility result.

1. **Spike detection is the real gap.** The model's edge vanishes when
   volatility rises (+1.8% vs +18.3%), and that is the regime a risk tool is
   for. Jump-robust estimators (bipower variation), implied volatility from the
   NIFTY options chain, and asymmetric targets that separate upside from
   downside realised vol are the candidates. Whether any of them help is an
   open question — vol jumps are hard for a reason.
2. **Position sizing, end to end.** Turn the forecast into what it is actually
   for: inverse-volatility weights on a paper portfolio, benchmarked against
   equal weighting. This is where the forecast either earns its keep or does
   not, and it is a fairer test than RMSE.
3. **Forward test.** Publish the monthly forecast and score it as the months
   arrive, so the repo eventually reports genuine out-of-sample results rather
   than a held-out split.

For the factor side, the known gap is **fundamentals**. With daily bars alone
there is no size, value, or quality factor, because the bhavcopy carries no
share count, book value, or earnings — and value and quality are the two
factors most associated with the long-horizon investing this was built for.
Separately, the book ran 22–23% volatility against the index's 14.5%, mostly an
uncontrolled small-cap tilt; sector and size neutralisation would show whether
the ranking has an edge once that tilt is removed, or whether it *was* the tilt.

## Not investment advice

A research tool for a paper-trading account. Nothing here is a recommendation,
and the author is not a registered investment adviser. Distributing signals
from something like this in India requires SEBI Research Analyst registration.

## License

MIT
