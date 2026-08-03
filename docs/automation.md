# Keeping the forecast current

The model predicts volatility 21 trading days ahead from daily bars. Feeding it
live intraday ticks would not make it more accurate, only noisier, so the
correct refresh rate is **once per day, after NSE publishes the bhavcopy**
(around 18:00 IST). Matching update frequency to forecast horizon is the actual
design decision here; anything faster is decoration.

## Daily run

```bash
python scripts/daily_update.py
```

It fetches only sessions newer than the stored panel, appends them, reuses the
persisted model, regenerates the forecasts and the page. On a warm cache the
whole thing takes under a minute. With no new session — a weekend, a holiday —
it exits early having done nothing.

```bash
python scripts/daily_update.py --check-only   # freshness only, no fetching
python scripts/daily_update.py --retrain      # force a model refit
```

## Why it fails loudly

A page showing last month's forecast looks exactly like a page showing today's.
Silence is therefore the dangerous outcome, so `daily_update.py` exits non-zero
when the newest session is more than five calendar days old rather than
publishing what happens to be on disk. Five days absorbs a weekend plus a
public holiday without false alarms.

This is the same principle as the 403-versus-404 handling in `bhavcopy.py`:
missing data must never be able to masquerade as normal.

## Scheduling

Two options, and which one you need depends on a question that has to be
answered empirically.

### GitHub Actions (preferred)

`.github/workflows/daily.yml` runs weekdays at 13:30 UTC (19:00 IST) and
`retrain.yml` refits monthly. Nothing needs to be switched on at home.

**The open question is whether NSE serves GitHub's runners at all.** NSE
throttles aggressively and treats datacenter IP ranges more harshly than
residential ones — this project has already been IP-banned once from a laptop
for fetching too fast. The workflow therefore probes `nsearchives.nseindia.com`
as its first step and fails with an explicit message if blocked, so the
diagnosis is unambiguous rather than surfacing three steps later as a confusing
download error.

Verify by pushing the repo and running the workflow manually
(**Actions → Daily volatility refresh → Run workflow**). If the probe fails,
use the local option below.

The market-data cache is load-bearing rather than an optimisation: without it
each run would re-download ~2,900 sessions and be banned within a day.

### Local schedule (fallback)

On macOS, a `launchd` agent at `~/Library/LaunchAgents/com.nsefactor.daily.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.nsefactor.daily</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/anoopibrampur/nsefactor/.venv/bin/python</string>
    <string>scripts/daily_update.py</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/anoopibrampur/nsefactor</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>19</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>/tmp/nsefactor-daily.log</string>
  <key>StandardErrorPath</key><string>/tmp/nsefactor-daily.err</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.nsefactor.daily.plist
```

The trade-off is real: this only runs when the machine is awake and online, so
the page can silently fall behind. The freshness guard limits the damage — the
run fails rather than republishing stale numbers — but nobody is watching the
log. Prefer Actions if the probe passes.

## What is not automated, and why

**Fundamentals.** NSE stopped populating the financial-results endpoint after
its March 2025 migration to API-based filing, so there is nothing new to fetch.
Automating it would poll an endpoint that no longer answers.

**Retraining on the daily path.** Volatility dynamics do not turn over
overnight. Refitting every run would also slide the chronological train/test
split forward daily, moving the published headline figures for no real reason —
which reads as instability to anyone watching the page. Monthly is the cadence.

## Publishing

The daily job uploads `artifacts/demo/index.html` as a workflow artifact. To
serve it publicly, add a GitHub Pages deploy step — the page is fully
self-contained, with no external requests, so it needs no build step and no
CDN.
