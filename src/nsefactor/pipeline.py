"""Incremental daily update: fetch new sessions, refresh forecasts, guard staleness.

The research scripts in this repo rebuild everything from scratch every run,
which is right for research and wrong for a scheduled job. Three changes make
a daily refresh cheap:

* **Append, don't rebuild.** ``load_cached()`` re-parses all ~2,900 bhavcopy
  zips to produce the panel. Only the new sessions need parsing.
* **Cache the historical universe.** Resolving membership across every past
  rebalance date dominates the runtime, and none of it changes when a new day
  arrives. Only today's universe is new.
* **Separate inference from training.** Volatility dynamics do not shift
  overnight, so the model is refit on its own schedule and loaded for daily
  inference. This also keeps the published numbers stable: refitting every run
  slides the chronological split forward and nudges the headline figures, which
  reads as instability to anyone watching the page.

The staleness guard exists because the dangerous failure is silent. A page
showing last month's forecast looks exactly like a page showing today's, so
this module treats "data older than expected" as an error rather than letting
it publish quietly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from . import bhavcopy as bc
from .config import DATA_DIR

log = logging.getLogger(__name__)

PANEL_PATH = DATA_DIR / "bhavcopy.parquet"
MODEL_PATH = DATA_DIR.parent / "models" / "vol_model.joblib"


class StaleDataError(RuntimeError):
    """The panel is older than a scheduled refresh should ever leave it."""


@dataclass
class UpdateResult:
    sessions_added: int
    latest_session: pd.Timestamp
    total_sessions: int
    rows: int


def append_new_sessions(
    start_hint: str | None = None,
    end: str | None = None,
    max_workers: int = 2,
) -> UpdateResult:
    """Fetch sessions newer than the stored panel and append them.

    Only days after the panel's last session are requested, and NSE's 404 for a
    holiday is not an error. Returns what changed so a caller can decide
    whether any downstream work is needed at all.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    if PANEL_PATH.exists():
        existing = pd.read_parquet(PANEL_PATH)
        last = pd.Timestamp(existing["date"].max())
        start = start_hint or (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        existing = None
        start = start_hint or "2015-01-01"
        last = None

    if last is not None and pd.Timestamp(start) > pd.Timestamp(end):
        log.info("panel already current through %s", last.date())
        days = bc.trading_days(existing)
        return UpdateResult(0, last, len(days), len(existing))

    log.info("fetching sessions %s -> %s", start, end)
    fresh = bc.load_range(start, end, max_workers=max_workers, strict=False)

    if fresh.empty:
        if existing is None:
            raise RuntimeError("no data fetched and no existing panel")
        days = bc.trading_days(existing)
        log.info("no new sessions; panel stands at %s", last.date())
        return UpdateResult(0, last, len(days), len(existing))

    if existing is None:
        combined = fresh
        added = fresh["date"].nunique()
    else:
        known = set(existing["date"].unique())
        added = int(fresh.loc[~fresh["date"].isin(known), "date"].nunique())
        # Concatenate everything fetched rather than only strictly newer dates.
        # NSE does republish a corrected bhavcopy for a session it has already
        # released, and filtering on `date > last` drops those corrections
        # silently -- the panel keeps the superseded bar and nothing complains.
        combined = pd.concat([existing, fresh], ignore_index=True)

    # Fresh rows are concatenated last, so keeping the last occurrence lets a
    # re-fetched session overwrite the stored one instead of duplicating it.
    combined = combined.drop_duplicates(subset=["date", "isin"], keep="last")
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PANEL_PATH, index=False)

    days = bc.trading_days(combined)
    log.info("appended %d sessions; panel now %s", added, days[-1].date())
    return UpdateResult(added, days[-1], len(days), len(combined))


def assert_fresh(
    latest_session: pd.Timestamp,
    now: pd.Timestamp | None = None,
    max_age_days: int = 5,
) -> None:
    """Raise if the newest session is older than a working refresh would allow.

    Five calendar days absorbs a weekend plus a public holiday without firing,
    while still catching a feed that has genuinely stopped. The alternative --
    publishing whatever is on disk -- produces a page that looks current and
    is not, which is the failure this whole module exists to prevent.
    """
    now = now or pd.Timestamp.today().normalize()
    age = (now.normalize() - pd.Timestamp(latest_session).normalize()).days
    if age > max_age_days:
        raise StaleDataError(
            f"latest session {pd.Timestamp(latest_session).date()} is {age} days old "
            f"(limit {max_age_days}). Refusing to publish stale data."
        )
    log.info("data freshness OK: %d day(s) old", age)


def save_model(model, path=MODEL_PATH, metadata: dict | None = None) -> None:
    """Persist a fitted model plus the metadata needed to judge it later."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata or {}}, path)
    log.info("wrote model to %s", path)


def load_model(path=MODEL_PATH):
    """Load a persisted model. Returns ``(model, metadata)`` or ``(None, {})``."""
    import joblib

    if not path.exists():
        return None, {}
    blob = joblib.load(path)
    return blob.get("model"), blob.get("metadata", {})


def model_age_days(metadata: dict, now: pd.Timestamp | None = None) -> float | None:
    """How long since the persisted model was fitted."""
    trained = metadata.get("trained_on")
    if not trained:
        return None
    now = now or pd.Timestamp.today()
    return float((now.normalize() - pd.Timestamp(trained).normalize()).days)
