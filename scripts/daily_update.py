"""Daily refresh: fetch new sessions, rescore, rebuild the page.

Usage:
    python scripts/daily_update.py              # normal daily run
    python scripts/daily_update.py --retrain     # refit the model as well
    python scripts/daily_update.py --check-only  # freshness check, no work

Designed to be run unattended, which changes the priorities from the research
scripts:

* it fetches only sessions newer than the stored panel;
* it loads a persisted model rather than refitting, so the published numbers
  do not drift every run and the job finishes in seconds rather than minutes;
* it exits non-zero on stale data instead of publishing it.

That last point is the one that matters. A page showing last month's forecast
is indistinguishable from one showing today's, so silence is the dangerous
outcome and the job is written to fail loudly instead.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

import pandas as pd

from nsefactor import pipeline
from nsefactor.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("daily")

# Refit roughly monthly. Volatility dynamics do not turn over faster than that,
# and refitting on the daily path would slide the chronological split forward
# every run, moving the headline figures for no real reason.
RETRAIN_AFTER_DAYS = 30


def run(script: str) -> None:
    log.info("running %s", script)
    result = subprocess.run(
        [sys.executable, f"scripts/{script}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("%s failed:\n%s", script, result.stdout[-3000:] + result.stderr[-3000:])
        raise SystemExit(f"{script} failed with code {result.returncode}")
    tail = [ln for ln in result.stdout.strip().splitlines() if ln.strip()][-3:]
    for ln in tail:
        log.info("  %s", ln)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true",
                    help="refit the volatility model regardless of its age")
    ap.add_argument("--check-only", action="store_true",
                    help="report freshness and exit without fetching")
    ap.add_argument("--max-age-days", type=int, default=5,
                    help="fail if the newest session is older than this")
    args = ap.parse_args()

    if args.check_only:
        if not pipeline.PANEL_PATH.exists():
            log.error("no panel at %s", pipeline.PANEL_PATH)
            return 1
        panel = pd.read_parquet(pipeline.PANEL_PATH, columns=["date"])
        latest = pd.Timestamp(panel["date"].max())
        try:
            pipeline.assert_fresh(latest, max_age_days=args.max_age_days)
        except pipeline.StaleDataError as exc:
            log.error("%s", exc)
            return 1
        log.info("panel current through %s", latest.date())
        return 0

    # ---- 1. Data ---------------------------------------------------------
    result = pipeline.append_new_sessions()
    log.info(
        "panel: %s sessions, latest %s (%d added)",
        f"{result.total_sessions:,}", result.latest_session.date(), result.sessions_added,
    )

    try:
        pipeline.assert_fresh(result.latest_session, max_age_days=args.max_age_days)
    except pipeline.StaleDataError as exc:
        log.error("%s", exc)
        return 1

    if result.sessions_added == 0:
        log.info("no new sessions; nothing to rebuild")
        return 0

    # ---- 2. Model --------------------------------------------------------
    _, meta = pipeline.load_model()
    age = pipeline.model_age_days(meta)
    if args.retrain or age is None or age > RETRAIN_AFTER_DAYS:
        why = "forced" if args.retrain else ("absent" if age is None else f"{age:.0f} days old")
        log.info("model refit needed (%s)", why)
    else:
        log.info("model is %.0f days old; reusing", age)

    # ---- 3. Forecasts and page ------------------------------------------
    run("vol_demo.py")
    run("build_demo_page.py")

    page = DATA_DIR.parent / "demo" / "index.html"
    log.info("page rebuilt: %s (%s bytes)", page, f"{page.stat().st_size:,}")
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
