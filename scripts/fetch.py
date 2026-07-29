"""Download the full bhavcopy history and write a normalised parquet panel.

Usage:  python scripts/fetch.py [START] [END]

Re-running is cheap: raw zips are cached under artifacts/raw, so only days
that were never fetched hit the network.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from nsefactor import bhavcopy as bc
from nsefactor.config import DATA_DIR, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch")


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG.start_date
    end = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.today().strftime("%Y-%m-%d")

    log.info("fetching bhavcopy %s -> %s", start, end)
    panel = bc.load_range(start, end)

    if panel.empty:
        log.error("no data fetched")
        raise SystemExit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "bhavcopy.parquet"
    panel.to_parquet(out, index=False)

    days = bc.trading_days(panel)
    log.info(
        "wrote %s: %d rows, %d trading days, %s -> %s, %d unique ISINs",
        out,
        len(panel),
        len(days),
        days[0].date(),
        days[-1].date(),
        panel["isin"].nunique(),
    )


if __name__ == "__main__":
    main()
