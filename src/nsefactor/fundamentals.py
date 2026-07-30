"""Point-in-time fundamentals from NSE's XBRL financial-result filings.

Everything in this module exists to answer one question honestly: **what did a
person standing on date T actually know about this company's finances?**

Getting that wrong is the standard way a value or quality factor produces a
spectacular backtest and no real edge. A company's March-quarter earnings are
not public in March -- Indian filers have 45 days for a quarter and 60 for the
final one, and delinquent filers take far longer. One filing in this dataset
reports the June-2024 quarter with a broadcast date of May 2026, nearly two
years late. Any factor that used those numbers before they existed is reading
the future.

So every record here is stamped with ``broadcast_date``, taken from NSE's own
``broadCastDate`` field, and :func:`as_of` is the only sanctioned way to query
the panel.

Sourcing
--------
Two hosts, two behaviours:

* ``www.nseindia.com/api/corporates-financial-results`` lists filings with
  metadata, including the broadcast timestamp, the ISIN, and a link to the
  XBRL document. It returns 403 to a bare request and needs a cookie from the
  homepage first -- the same handshake a browser performs.
* ``nsearchives.nseindia.com/corporate/xbrl/...`` serves the XBRL documents
  themselves on a plain GET, no cookie required.

Coverage runs from 2015 to roughly March 2025, after which NSE migrated to an
API-based single-filing system and this endpoint stopped being populated. That
is ample for a backtest; a live screen needs a current-snapshot source and
should say so.

A trap worth naming
-------------------
In these filings the XBRL ``contextRef`` periods are not trustworthy. A Q2
document labels both its quarterly context (``OneD``) and its year-to-date
context (``FourD``) with the *same* start and end dates. The true period lives
in the ``DateOfStartOfReportingPeriod`` and ``DateOfEndOfReportingPeriod``
facts reported inside each context. Reading the context dates instead makes a
six-month figure look like a three-month one, which roughly doubles revenue
and earnings for every company that files cumulatively.
"""

from __future__ import annotations

import calendar
import json
import logging
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import requests

from .config import ARTIFACTS_DIR, USER_AGENT

log = logging.getLogger(__name__)

NSE_HOME = "https://www.nseindia.com/"
RESULTS_API = "https://www.nseindia.com/api/corporates-financial-results"
RESULTS_REFERER = (
    "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
)

INDEX_DIR = ARTIFACTS_DIR / "fundamentals" / "index"
XBRL_DIR = ARTIFACTS_DIR / "fundamentals" / "xbrl"
FUND_DIR = ARTIFACTS_DIR / "fundamentals"

# The last month with meaningful filing volume. Beyond this the endpoint
# returns a handful of stragglers rather than the market.
COVERAGE_END = "2025-03"

# --- XBRL tags we care about, mapped to our own names ----------------------
# Flow items, reported for a period.
FLOW_TAGS = {
    "revenue": "RevenueFromOperations",
    "other_income": "OtherIncome",
    "pbt": "ProfitBeforeTax",
    "tax": "TaxExpense",
    "pat": "ProfitLossForPeriod",
    "depreciation": "DepreciationDepletionAndAmortisationExpense",
    "finance_cost": "FinanceCosts",
    "paid_up_capital": "PaidUpValueOfEquityShareCapital",
    "face_value": "FaceValueOfEquityShareCapital",
    # Operating cash flow enables an accruals factor: earnings unsupported by
    # cash are the classic signal of low-quality profit. Only annual filings
    # usually carry it, so expect it to be sparse.
    "cfo": "CashFlowsFromUsedInOperatingActivities",
}

# Stock items, reported at an instant (balance-sheet date).
STOCK_TAGS = {
    "share_capital": "EquityShareCapital",
    "other_equity": "OtherEquity",
    "borrowings_noncurrent": "BorrowingsNoncurrent",
    "borrowings_current": "BorrowingsCurrent",
}


@dataclass
class Filing:
    """One company-period filing, with the date it became public."""

    isin: str
    symbol: str
    broadcast_date: pd.Timestamp
    period_start: pd.Timestamp | None
    period_end: pd.Timestamp | None
    consolidated: bool
    audited: bool
    xbrl_url: str
    seq: str


# ---------------------------------------------------------------------------
# Filing index (metadata)
# ---------------------------------------------------------------------------


def _session() -> requests.Session:
    """A session carrying the cookie the results API requires.

    The homepage itself answers 403, but it still sets the cookie the API
    checks, so the handshake works despite the status code. Treating the 403
    as fatal would wrongly conclude the endpoint is unavailable.
    """
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    try:
        s.get(NSE_HOME, timeout=30)
    except Exception as exc:  # pragma: no cover - network
        log.warning("homepage handshake failed: %s", exc)
    s.headers.update({"Accept": "*/*", "Referer": RESULTS_REFERER})
    return s


def fetch_index_month(
    year: int,
    month: int,
    period: str = "Quarterly",
    session: requests.Session | None = None,
    pause: float = 1.2,
) -> list[dict]:
    """Filing metadata broadcast within one calendar month, cached to disk."""
    path = INDEX_DIR / f"{year}" / f"index_{year}{month:02d}_{period}.json"
    if path.exists():
        return json.loads(path.read_text())

    s = session or _session()
    last = calendar.monthrange(year, month)[1]
    url = (
        f"{RESULTS_API}?index=equities&period={period}"
        f"&from_date=01-{month:02d}-{year}&to_date={last}-{month:02d}-{year}"
    )
    time.sleep(pause)
    resp = s.get(url, timeout=90)
    resp.raise_for_status()
    text = resp.text.strip()
    records = json.loads(text) if text.startswith("[") else []

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))
    return records


def _parse_broadcast(value: str) -> pd.Timestamp | None:
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
        try:
            return pd.Timestamp(pd.to_datetime(value, format=fmt))
        except (ValueError, TypeError):
            continue
    return None


def _parse_day(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.Timestamp(pd.to_datetime(value, format="%d-%b-%Y"))
    except (ValueError, TypeError):
        return None


def index_to_filings(records: list[dict]) -> list[Filing]:
    """Convert raw API records into :class:`Filing` objects, dropping unusable ones."""
    out = []
    for r in records:
        isin = (r.get("isin") or "").strip()
        url = r.get("xbrl") or ""
        bc = _parse_broadcast(r.get("broadCastDate", ""))
        if not isin or not url or bc is None:
            continue
        out.append(
            Filing(
                isin=isin,
                symbol=(r.get("symbol") or "").strip(),
                broadcast_date=bc,
                period_start=_parse_day(r.get("fromDate")),
                period_end=_parse_day(r.get("toDate")),
                consolidated=(r.get("consolidated") == "Consolidated"),
                audited=(r.get("audited") == "Audited"),
                xbrl_url=url,
                seq=str(r.get("seqNumber", "")),
            )
        )
    return out


def build_filing_index(
    start: str = "2015-01",
    end: str = COVERAGE_END,
    period: str = "Quarterly",
) -> pd.DataFrame:
    """Assemble the full filing index across a month range."""
    months = pd.period_range(start, end, freq="M")
    session = _session()
    rows: list[Filing] = []
    for p in months:
        try:
            recs = fetch_index_month(p.year, p.month, period, session)
        except Exception as exc:
            log.warning("index fetch failed for %s: %s", p, exc)
            continue
        rows.extend(index_to_filings(recs))
        log.info("%s: %d filings", p, len(recs))

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([f.__dict__ for f in rows])
    return df.sort_values(["isin", "broadcast_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# XBRL documents
# ---------------------------------------------------------------------------


def download_xbrl(url: str, pause: float = 0.0) -> bytes | None:
    """Fetch one XBRL document, cached by filename. No cookie needed."""
    name = url.rstrip("/").split("/")[-1]
    if not name.lower().endswith(".xml"):
        name += ".xml"
    path = XBRL_DIR / name[:4] / name
    if path.exists():
        return path.read_bytes()

    if pause:
        time.sleep(pause)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    except Exception as exc:
        log.debug("xbrl fetch error %s: %s", name, exc)
        return None
    if resp.status_code != 200 or not resp.content:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return resp.content


def parse_xbrl(blob: bytes) -> dict | None:
    """Extract one quarter's figures from an XBRL document.

    Returns the *quarterly* context only. Where a filer reports cumulatively,
    the year-to-date context is identified by its true reporting dates and
    excluded, because summing it with quarterly figures would double-count.
    """
    try:
        root = ET.fromstring(blob)
    except ET.ParseError:
        return None

    # Instant contexts, for balance-sheet items.
    instants: dict[str, str] = {}
    for c in root.iter():
        if not c.tag.endswith("}context"):
            continue
        cid = c.get("id")
        for x in c.iter():
            if x.tag.endswith("}instant") and x.text:
                instants[cid] = x.text.strip()

    facts: dict[str, dict[str, str]] = defaultdict(dict)
    for el in root:
        tag = el.tag.split("}")[-1]
        ref = el.get("contextRef")
        if ref and el.text and el.text.strip():
            facts[tag][ref] = el.text.strip()

    # True period per context, from the reported dates rather than the
    # contextRef, which is unreliable in these documents.
    starts = facts.get("DateOfStartOfReportingPeriod", {})
    ends = facts.get("DateOfEndOfReportingPeriod", {})
    if not starts or not ends:
        return None

    spans = {}
    for ref, sd in starts.items():
        ed = ends.get(ref)
        if not ed:
            continue
        try:
            s, e = pd.Timestamp(sd), pd.Timestamp(ed)
        except (ValueError, TypeError):
            continue
        spans[ref] = (s, e, (e - s).days)

    if not spans:
        return None

    # The quarter is the shortest span that still looks like a quarter.
    quarterly = {r: v for r, v in spans.items() if 60 <= v[2] <= 100}
    chosen = min(quarterly.items(), key=lambda kv: kv[1][2]) if quarterly else None
    if chosen is None:
        return None
    qref, (qstart, qend, _) = chosen

    def num(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).replace(",", ""))
        except ValueError:
            return None

    out: dict = {
        "period_start": qstart,
        "period_end": qend,
        "quarter_days": (qend - qstart).days,
    }
    for name, tag in FLOW_TAGS.items():
        out[name] = num(facts.get(tag, {}).get(qref))

    # Balance-sheet items sit on the instant context closest to the quarter
    # end, which is normally the quarter end itself.
    best_ref, best_gap = None, None
    for ref, day in instants.items():
        try:
            gap = abs((pd.Timestamp(day) - qend).days)
        except (ValueError, TypeError):
            continue
        if best_gap is None or gap < best_gap:
            best_ref, best_gap = ref, gap
    for name, tag in STOCK_TAGS.items():
        out[name] = num(facts.get(tag, {}).get(best_ref)) if best_ref else None
    out["balance_sheet_date"] = pd.Timestamp(instants[best_ref]) if best_ref else None

    # Derived quantities.
    pu, fv = out.get("paid_up_capital"), out.get("face_value")
    out["shares_outstanding"] = (pu / fv) if (pu and fv and fv > 0) else None

    sc, oe = out.get("share_capital"), out.get("other_equity")
    out["net_worth"] = (sc or 0) + (oe or 0) if (sc is not None or oe is not None) else None

    bn, bc = out.get("borrowings_noncurrent"), out.get("borrowings_current")
    out["total_debt"] = (bn or 0) + (bc or 0) if (bn is not None or bc is not None) else None

    return out


# ---------------------------------------------------------------------------
# Point-in-time access
# ---------------------------------------------------------------------------


def as_of(panel: pd.DataFrame, when: pd.Timestamp, max_age_days: int = 400) -> pd.DataFrame:
    """The latest filing per ISIN that was public on ``when``.

    Three rules, each of which matters:

    * only filings with ``broadcast_date <= when`` are visible;
    * among those, the most recently *broadcast* wins, which naturally
      handles revisions and restatements -- a corrected filing supersedes the
      original from its own broadcast date onward, not retroactively;
    * filings whose period ended more than ``max_age_days`` before ``when``
      are dropped as stale, so a company that stopped reporting does not keep
      contributing year-old fundamentals indefinitely.
    """
    visible = panel[panel["broadcast_date"] <= when]
    if visible.empty:
        return visible

    fresh = visible[(when - visible["period_end"]).dt.days <= max_age_days]
    if fresh.empty:
        return fresh

    ordered = fresh.sort_values(["isin", "broadcast_date", "period_end"])
    return ordered.groupby("isin", as_index=False).tail(1).set_index("isin")


def trailing_four_quarters(
    panel: pd.DataFrame, when: pd.Timestamp, column: str = "pat"
) -> pd.Series:
    """Sum ``column`` over the four most recent distinct quarters visible at ``when``.

    Quarterly earnings are seasonal -- a single quarter annualised is noise for
    most Indian businesses -- so every flow-based factor uses a trailing year.
    Deduplicating on ``period_end`` first ensures a revision does not get
    counted twice alongside the filing it replaces.
    """
    visible = panel[panel["broadcast_date"] <= when]
    if visible.empty:
        return pd.Series(dtype=float)

    ordered = visible.sort_values(["isin", "period_end", "broadcast_date"])
    latest = ordered.groupby(["isin", "period_end"], as_index=False).tail(1)

    out = {}
    for isin, grp in latest.groupby("isin", sort=False):
        recent = grp.nlargest(4, "period_end")
        if len(recent) < 4 or recent[column].isna().any():
            continue
        # Require the four quarters to actually span about a year; a company
        # with gaps would otherwise sum four scattered quarters.
        span = (recent["period_end"].max() - recent["period_end"].min()).days
        if not (240 <= span <= 460):
            continue
        out[isin] = float(recent[column].sum())
    return pd.Series(out, name=f"ttm_{column}")
