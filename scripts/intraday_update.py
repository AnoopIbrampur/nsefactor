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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="refresh even when the market is closed")
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
    if not is_open and not args.force:
        log.info("market closed at %s; leaving the live panel as it stands",
                 now.strftime("%Y-%m-%d %H:%M %Z"))
        return 0

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
        payload = {
            "as_of": now.isoformat(),
            "last_bar": str(bars.index[-1]),
            "market_open": is_open,
            "available": True,
            "scored": len(rows),
            "hot": sum(1 for r in rows if r["surprise"] >= 2.5),
            "rows": rows[:TOP_N],
        }

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out = SITE_DIR / "live.json"
    out.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
