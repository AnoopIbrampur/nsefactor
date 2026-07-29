"""NSE index levels, used as the bar the strategy has to clear.

Same archive host and caching discipline as :mod:`bhavcopy`. One CSV per day
holds the close for all ~160 published indices.

NSE publishes only *price* indices here, not total-return ones. A price index
excludes dividends, so comparing a dividend-inclusive strategy against it
flatters the strategy by roughly the market's yield (~1.2%/yr for the Nifty
50). :func:`total_return_proxy` accrues the published yield back in so the
comparison can be made on both bases.
"""

from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import pandas as pd
import requests

from .bhavcopy import Throttled
from .config import ARCHIVE_HOST, RAW_DIR, USER_AGENT

log = logging.getLogger(__name__)

INDEX_DIR = RAW_DIR.parent / "indices"

COLUMNS = ["date", "index_name", "open", "high", "low", "close", "pe", "pb", "div_yield"]

_RENAME = {
    "Index Name": "index_name",
    "Open Index Value": "open",
    "High Index Value": "high",
    "Low Index Value": "low",
    "Closing Index Value": "close",
    "P/E": "pe",
    "P/B": "pb",
    "Div Yield": "div_yield",
}


def _url_for(day: date) -> str:
    return f"{ARCHIVE_HOST}/content/indices/ind_close_all_{day:%d%m%Y}.csv"


def fetch_raw(day: date, session: requests.Session | None = None, retries: int = 5) -> bytes | None:
    """Fetch one day's index CSV, caching to disk. ``None`` if no session."""
    path = INDEX_DIR / f"{day:%Y}" / f"idx_{day:%Y%m%d}.csv"
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
                time.sleep(delay)
                delay *= 2
                continue
            raise Throttled(f"{resp.status_code} for index {day}")
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        return resp.content
    raise Throttled(f"exhausted retries for index {day}")


def parse(blob: bytes, day: date) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(blob))
    df = df.rename(columns=_RENAME)
    df["date"] = pd.Timestamp(day)
    df["index_name"] = df["index_name"].astype(str).str.strip()
    for col in ("open", "high", "low", "close", "pe", "pb", "div_yield"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")
    return df[COLUMNS]


def load_range(start, end, max_workers: int = 2, strict: bool = False) -> pd.DataFrame:
    """Fetch index levels across a date range, mirroring bhavcopy semantics."""
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    days = [d.date() for d in pd.date_range(start, end, freq="D")]

    session = requests.Session()
    frames: list[pd.DataFrame] = []
    failed: list[date] = []

    def one(day: date):
        try:
            blob = fetch_raw(day, session)
        except Throttled:
            return day
        except Exception as exc:
            log.warning("index fetch failed %s: %s", day, exc)
            return day
        return None if blob is None else parse(blob, day)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(one, days):
            if result is None:
                continue
            if isinstance(result, date):
                failed.append(result)
            else:
                frames.append(result)

    log.info("index: %d days loaded, %d failures", len(frames), len(failed))
    if failed and strict:
        raise Throttled(f"{len(failed)} index days failed, first: {failed[:5]}")
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "index_name"])


def series(index_data: pd.DataFrame, name: str = "Nifty 50") -> pd.DataFrame:
    """Single index as a date-indexed frame with daily price returns."""
    one = index_data[index_data["index_name"] == name].sort_values("date")
    if one.empty:
        raise KeyError(f"index {name!r} not found; available e.g. {sorted(index_data['index_name'].unique())[:5]}")
    out = one.set_index("date")[["close", "div_yield"]].copy()
    out["ret"] = out["close"].pct_change()
    return out


def total_return_proxy(index_series: pd.DataFrame) -> pd.Series:
    """Approximate total return by accruing the published dividend yield daily.

    NSE reports ``div_yield`` as an annual percentage for the index on each
    date. Spreading it across 252 trading days and adding it to the price
    return gives a total-return approximation. It is an approximation --
    dividends arrive lumpily on ex-dates, not smoothly -- but over multi-year
    horizons the compounding difference is small, and it removes a systematic
    ~1.2%/yr handicap that would otherwise be handed to the strategy.
    """
    daily_yield = index_series["div_yield"].fillna(0.0) / 100.0 / 252.0
    return index_series["ret"] + daily_yield
