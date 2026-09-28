"""Refresh the live panel: today's moves against the standing forecast.

Usage:
    python scripts/intraday_update.py            # refresh if the market is open
    python scripts/intraday_update.py --force    # refresh regardless

Writes ``artifacts/site/live.json``, which the published page fetches from its
own origin. Same-origin means no CORS negotiation and no external request from
the browser, so the page stays a plain static file.

This deliberately does not re-run the model. The forecast covers 21 trading
days and does not move between fifteen-minute refreshes; what changes is
whether today is behaving as predicted.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from nsefactor import intraday
from nsefactor.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("intraday")

SITE_DIR = DATA_DIR.parent / "site"
DEMO_JSON = DATA_DIR.parent / "reports" / "vol_demo.json"

# Cap the live panel. Fetching every name is cheap, but the page only needs the
# extremes to be useful and a 500-row live table is noise.
TOP_N = 25

# A quote older than this is not a live view. Four days absorbs a weekend plus a
# public holiday without tripping.
STALE_AFTER_DAYS = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="deprecated and ignored; every run refreshes now")
    ap.add_argument("--limit", type=int, default=None,
                    help="only fetch the first N symbols (for a quick check)")
    args = ap.parse_args()

    if not DEMO_JSON.exists():
        log.error("no forecasts at %s; run scripts/vol_demo.py first", DEMO_JSON)
        return 1

    demo = json.loads(DEMO_JSON.read_text())
    forecasts = demo["forecasts"]
    if args.limit:
        forecasts = forecasts[: args.limit]

    now = pd.Timestamp.now(tz=intraday.MARKET_TZ)
    is_open = intraday.market_is_open(now)
    # Refresh even when the market is closed.
    #
    # This used to return early outside trading hours, on the reasoning that
    # there was nothing new to fetch. The effect was worse than the saving: the
    # panel kept whatever it last wrote and went on serving it indefinitely, so
    # after a run of closed-market ticks it was showing a session seven weeks
    # old under a "market closed" label that made it look current. Outside
    # trading hours the feed still returns the last completed session, which is
    # the honest thing to show, and `market_open` says which it is.
    log.info("market %s at %s", "open" if is_open else "closed",
             now.strftime("%Y-%m-%d %H:%M %Z"))

    symbols = [f["symbol"] for f in forecasts]
    log.info("fetching %d symbols", len(symbols))
    bars = intraday.fetch_intraday(symbols)
    prev = intraday.previous_closes(symbols)

    if bars.empty or prev.empty:
        # A failed live fetch is not a reason to fail the job: the forecast page
        # is still valid without a live panel, and marking the data stale is
        # more honest than emitting an empty table that looks like calm markets.
        log.warning("no intraday data available; writing an explicit gap marker")
        payload = {
            "as_of": now.isoformat(),
            "market_open": is_open,
            "available": False,
            "reason": "intraday feed returned nothing",
            "rows": [],
        }
    else:
        rows = intraday.surprise_table(forecasts, bars, prev)
        log.info("scored %d symbols; most surprising: %s",
                 len(rows),
                 ", ".join(f"{r['symbol']} {r['surprise']:.1f}x" for r in rows[:3]))
        last_bar = bars.index[-1]
        age = intraday.session_age_days(last_bar, now)
        payload = {
            "as_of": now.isoformat(),
            "last_bar": str(last_bar),
            "session_age_days": age,
            "market_open": is_open,
            # Stated explicitly rather than inferred by the page. A quote from
            # last week is not a live view, however plausibly a "market closed"
            # label frames it, so the page hides the panel instead of dressing
            # stale numbers as the latest session.
            "available": age <= STALE_AFTER_DAYS,
            "stale": age > STALE_AFTER_DAYS,
            "scored": len(rows),
            "hot": sum(1 for r in rows if r["surprise"] >= 2.5),
            "rows": rows[:TOP_N],
        }
        if age > STALE_AFTER_DAYS:
            log.warning("newest quote is %d days old; marking the panel stale", age)

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out = SITE_DIR / "live.json"
    out.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
