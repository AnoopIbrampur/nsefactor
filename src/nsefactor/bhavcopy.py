"""Download and normalise NSE daily bhavcopy files.

NSE publishes end-of-day equity data in two incompatible layouts:

* legacy   (through 2023):  ``cm{DDMMMYYYY}bhav.csv.zip`` under
  ``/content/historical/EQUITIES/{YYYY}/{MON}/``
* UDiFF    (2024 onwards):  ``BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip``
  under ``/content/cm/``

Both carry ISIN, so they normalise to one schema without an external symbol
map. That matters: NSE symbols get reused after delistings, ISINs do not.
"""

from __future__ import annotations

import io
import logging
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import pandas as pd
import requests

from .config import ARCHIVE_HOST, DEFAULT_CONFIG, RAW_DIR, UDIFF_CUTOVER, USER_AGENT, Config

log = logging.getLogger(__name__)

# Normalised schema every loader returns.
COLUMNS = [
    "date",
    "symbol",
    "isin",
    "open",
    "high",
    "low",
    "close",
    "prevclose",
    "volume",
    "turnover",
    "trades",
]

_LEGACY_RENAME = {
    "SYMBOL": "symbol",
    "ISIN": "isin",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "PREVCLOSE": "prevclose",
    "TOTTRDQTY": "volume",
    "TOTTRDVAL": "turnover",
    "TOTALTRADES": "trades",
}

_UDIFF_RENAME = {
    "TckrSymb": "symbol",
    "ISIN": "isin",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "PrvsClsgPric": "prevclose",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades",
}


def _url_for(day: date) -> str:
    """Return the archive URL for ``day``, picking the era-appropriate layout."""
    if day.isoformat() >= UDIFF_CUTOVER:
        return f"{ARCHIVE_HOST}/content/cm/BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    month = f"{day:%b}".upper()
    stamp = f"{day:%d%b%Y}".upper()
    return (
        f"{ARCHIVE_HOST}/content/historical/EQUITIES/"
        f"{day:%Y}/{month}/cm{stamp}bhav.csv.zip"
    )


def _cache_path(day: date):
    return RAW_DIR / f"{day:%Y}" / f"bhav_{day:%Y%m%d}.zip"


class Throttled(RuntimeError):
    """NSE refused the request after every retry. Distinct from 'no such file'."""


def fetch_raw(
    day: date,
    session: requests.Session | None = None,
    retries: int = 5,
) -> bytes | None:
    """Fetch one day's zip, caching to disk. ``None`` if NSE has no file.

    A 404 means a weekend or trading holiday and is not an error -- we never
    rebuild the trading calendar ourselves, the presence of a bhavcopy *is*
    the calendar.

    A 403 means we are being throttled, which is a completely different thing
    and must never be recorded as a holiday. Conflating the two silently
    punches holes in the price history, and a backtest reads those holes as
    real gaps in trading rather than as missing data. We retry with backoff
    and raise :class:`Throttled` if the day genuinely cannot be fetched.
    """
    path = _cache_path(day)
    if path.exists():
        return path.read_bytes()

    sess = session or requests.Session()
    delay = 1.0
    for attempt in range(retries):
        resp = sess.get(_url_for(day), headers={"User-Agent": USER_AGENT}, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code in (403, 429) or resp.status_code >= 500:
            if attempt < retries - 1:
                time.sleep(delay + random.uniform(0, 0.5))
                delay *= 2
                continue
            raise Throttled(f"{resp.status_code} for {day} after {retries} attempts")
        resp.raise_for_status()

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        return resp.content

    raise Throttled(f"exhausted retries for {day}")


def parse(blob: bytes, day: date, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Normalise one day's zip into :data:`COLUMNS`, equities only."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = pd.read_csv(io.BytesIO(zf.read(name)))

    if "TckrSymb" in raw.columns:  # UDiFF
        raw = raw[raw["FinInstrmTp"] == "STK"]
        series = raw["SctySrs"]
        df = raw.rename(columns=_UDIFF_RENAME)
    else:  # legacy
        series = raw["SERIES"]
        df = raw.rename(columns=_LEGACY_RENAME)

    keep = series.isin(cfg.equity_series).to_numpy()
    df = df.loc[keep].copy()

    # ETFs and fund units live in the EQ series but are not company equity.
    # Their ISINs start INF rather than INE.
    df = df[df["isin"].astype(str).str.startswith(cfg.equity_isin_prefix)]

    df["date"] = pd.Timestamp(day)
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["isin"] = df["isin"].astype(str).str.strip()
    for col in ("open", "high", "low", "close", "prevclose", "volume", "turnover", "trades"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[COLUMNS].reset_index(drop=True)


def load_range(
    start: str | date,
    end: str | date,
    cfg: Config = DEFAULT_CONFIG,
    max_workers: int = 4,
    strict: bool = True,
) -> pd.DataFrame:
    """Fetch and normalise every trading day in ``[start, end]``.

    Downloads run in parallel but stay narrow -- NSE's archive host throttles
    aggressively above a handful of concurrent requests, and the on-disk
    cache makes a re-run nearly free anyway.

    With ``strict`` (the default), any day we could not fetch raises rather
    than quietly shrinking the panel. Holidays, which return a clean 404, are
    not failures and never trigger this.
    """
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    # Every calendar day, not just weekdays. NSE runs a ceremonial one-hour
    # Muhurat session on Diwali, which usually falls on a Saturday or Sunday,
    # and it is a real settled trading session with its own bhavcopy. Skipping
    # weekends silently drops it, and the next session's prevclose then refers
    # to a day absent from the panel -- which reads downstream as a phantom
    # corporate action on the first weekday after Diwali.
    days = [d.date() for d in pd.date_range(start, end, freq="D")]

    session = requests.Session()
    frames: list[pd.DataFrame] = []
    holidays = 0
    failed: list[date] = []

    def one(day: date):
        try:
            blob = fetch_raw(day, session)
        except Throttled:
            return day  # signal failure, distinct from a holiday
        except Exception as exc:
            log.warning("fetch failed for %s: %s", day, exc)
            return day
        return None if blob is None else parse(blob, day, cfg)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for day, result in zip(days, pool.map(one, days)):
            if result is None:
                holidays += 1
            elif isinstance(result, date):
                failed.append(result)
            else:
                frames.append(result)

    log.info(
        "loaded %d trading days, %d holidays (404), %d failures",
        len(frames),
        holidays,
        len(failed),
    )
    if failed:
        msg = f"{len(failed)} days could not be fetched, first few: {failed[:5]}"
        if strict:
            raise Throttled(msg)
        log.warning(msg)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])


def load_cached(cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Rebuild the panel from cached zips only, touching no network.

    Useful when NSE is throttling: parsing is deterministic, so the panel can
    always be regenerated from whatever has already been downloaded.
    """
    paths = sorted(RAW_DIR.rglob("bhav_*.zip"))
    frames = []
    for path in paths:
        day = datetime.strptime(path.stem.removeprefix("bhav_"), "%Y%m%d").date()
        try:
            frames.append(parse(path.read_bytes(), day, cfg))
        except (zipfile.BadZipFile, StopIteration) as exc:
            log.warning("corrupt cache entry %s: %s", path.name, exc)

    log.info("rebuilt %d trading days from cache", len(frames))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])


def missing_days(start: str | date, end: str | date, weekdays_only: bool = True) -> list[date]:
    """Calendar days in range with no cached file. A superset of holidays.

    Defaults to weekdays because that is the useful worklist for spotting
    gaps; pass ``weekdays_only=False`` to include the Diwali Muhurat
    sessions, which trade on weekends.
    """
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    have = {p.stem.removeprefix("bhav_") for p in RAW_DIR.rglob("bhav_*.zip")}
    return [
        d.date()
        for d in pd.date_range(start, end, freq="D")
        if (d.weekday() < 5 or not weekdays_only) and f"{d:%Y%m%d}" not in have
    ]


def trading_days(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """The observed NSE trading calendar, derived from the data itself."""
    return pd.DatetimeIndex(sorted(panel["date"].unique()))


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()
