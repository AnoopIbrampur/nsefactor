"""Build the point-in-time fundamentals panel from NSE XBRL filings.

Usage:
    python scripts/fetch_fundamentals.py --index-only     # metadata only, ~3 min
    python scripts/fetch_fundamentals.py --measure        # how many XBRL files are needed
    python scripts/fetch_fundamentals.py                  # full build

Stage 1 fetches filing metadata month by month -- cheap, ~120 requests, and it
tells us the broadcast date and XBRL link for every filing. Stage 2 downloads
XBRL documents only for ISINs that were ever in the investable universe, which
is the difference between roughly 40,000 files and 300,000.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from nsefactor import adjust, backtest, fundamentals as F, universe
from nsefactor.config import DATA_DIR, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fundamentals")

CFG = DEFAULT_CONFIG


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def universe_isins() -> set[str]:
    """Every ISIN that was in the liquid universe at any rebalance date."""
    panel = pd.read_parquet(DATA_DIR / "bhavcopy.parquet")
    panel = adjust.apply_isin_links(panel)
    days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rebals = backtest.month_end_dates(days)
    rebals = rebals[rebals >= days[300]]

    seen: set[str] = set()
    for dt in rebals:
        sel = universe.select(panel, dt, cfg=CFG)
        seen.update(sel.index)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--start", default="2015-01")
    ap.add_argument("--end", default=F.COVERAGE_END)
    ap.add_argument("--pause", type=float, default=0.15,
                    help="seconds between XBRL downloads")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel XBRL downloads; the archive host bans above ~8")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap XBRL downloads (for a trial run)")
    args = ap.parse_args()

    F.FUND_DIR.mkdir(parents=True, exist_ok=True)
    index_path = F.FUND_DIR / "filing_index.parquet"

    # ---- Stage 1: filing metadata ---------------------------------------
    if index_path.exists():
        idx = pd.read_parquet(index_path)
        log.info("loaded cached filing index: %d filings", len(idx))
    else:
        section("STAGE 1: FILING INDEX")
        idx = F.build_filing_index(args.start, args.end)
        if idx.empty:
            log.error("no filings retrieved")
            raise SystemExit(1)
        idx.to_parquet(index_path, index=False)
        log.info("wrote %s", index_path)

    section("FILING INDEX SUMMARY")
    print(f"filings            {len(idx):,}")
    print(f"unique ISINs       {idx['isin'].nunique():,}")
    print(f"broadcast range    {idx['broadcast_date'].min().date()} -> "
          f"{idx['broadcast_date'].max().date()}")
    print(f"consolidated       {idx['consolidated'].mean():.1%}")
    by_year = idx.groupby(idx["broadcast_date"].dt.year).size()
    print("\nfilings broadcast per year:")
    print(by_year.to_string())

    # Reporting lag: the whole reason point-in-time matters.
    lag = (idx["broadcast_date"] - idx["period_end"]).dt.days.dropna()
    print("\nreporting lag, period end -> broadcast (days):")
    print(lag.describe(percentiles=[0.25, 0.5, 0.9, 0.99]).round(1).to_string())
    print(f"\nfilings arriving >120 days after period end: {(lag > 120).mean():.1%}")
    print("Those are exactly the ones a naive backtest would use months early.")

    if args.index_only:
        return

    # ---- Scope: only ISINs we could ever have traded ---------------------
    section("STAGE 2 SCOPE")
    log.info("resolving universe membership (this takes a few minutes)...")
    keep = universe_isins()
    print(f"ISINs ever in the liquid universe : {len(keep):,}")

    wanted = idx[idx["isin"].isin(keep)].copy()
    print(f"filings for those ISINs           : {len(wanted):,}")

    # Deliberately NOT deduped to one filing per (ISIN, quarter).
    #
    # Keeping only the first broadcast would cut the download roughly in half,
    # but of those first broadcasts only ~12% are consolidated: most Indian
    # companies file standalone quarterly and consolidated annually, and the
    # standalone filing usually goes out first. Standalone accounts cover the
    # parent alone, which materially understates a conglomerate, so a panel
    # built from whichever arrived first would mix two different accounting
    # scopes across the cross-section -- making a standalone filer look smaller
    # and less profitable than the same company reporting consolidated.
    #
    # Fetching everything lets the panel prefer consolidated per (ISIN, quarter)
    # where it exists and fall back to standalone otherwise, and keeps the
    # revision history for point-in-time work later.
    dedup_count = len(
        wanted.sort_values(["isin", "period_end", "broadcast_date"])
        .groupby(["isin", "period_end"], as_index=False)
        .head(1)
    )
    print(f"  consolidated                    : {int(wanted['consolidated'].sum()):,} "
          f"({wanted['consolidated'].mean():.1%})")
    print(f"  distinct (ISIN, quarter) pairs   : {dedup_count:,}")
    print("  fetching all filings so consolidated can be preferred per quarter")

    already = sum(1 for _ in F.XBRL_DIR.rglob("*.xml")) if F.XBRL_DIR.exists() else 0
    print(f"XBRL already cached               : {already:,}")
    print(f"still to download                 : {max(0, len(wanted) - already):,}")
    est_min = (len(wanted) - already) * (args.pause + 0.35) / 60
    print(f"rough time at pause={args.pause}s          : {est_min:.0f} min")

    if args.measure:
        return

    # ---- Stage 2: download and parse ------------------------------------
    section("STAGE 2: XBRL DOWNLOAD AND PARSE")
    rows = []
    todo = wanted if args.limit is None else wanted.head(args.limit)
    failed = parsed = 0

    def fetch_and_parse(rec):
        blob = F.download_xbrl(rec.xbrl_url, pause=args.pause)
        if blob is None:
            return None
        got = F.parse_xbrl(blob)
        if got is None:
            return None
        got.update(
            {
                "isin": rec.isin,
                "symbol": rec.symbol,
                "broadcast_date": rec.broadcast_date,
                "consolidated": rec.consolidated,
                "audited": rec.audited,
            }
        )
        return got

    # Modest concurrency. The archive host tolerates a handful of parallel
    # requests and IP-bans well before a dozen, and the on-disk cache makes a
    # resumed run cheap, so there is no reason to push it.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, got in enumerate(
            pool.map(fetch_and_parse, todo.itertuples(index=False)), 1
        ):
            if got is None:
                failed += 1
            else:
                rows.append(got)
                parsed += 1
            if n % 2000 == 0:
                log.info("%d/%d  parsed=%d failed=%d", n, len(todo), parsed, failed)

    if not rows:
        log.error("nothing parsed")
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    # Prefer consolidated where a company files both for the same quarter:
    # consolidated reflects the group a shareholder actually owns.
    df = df.sort_values(["isin", "period_end", "consolidated", "broadcast_date"])
    out_path = F.FUND_DIR / "fundamentals.parquet"
    df.to_parquet(out_path, index=False)

    section("RESULT")
    print(f"parsed {parsed:,} filings, {failed:,} failures "
          f"({failed / max(1, parsed + failed):.1%})")
    print(f"unique ISINs   {df['isin'].nunique():,}")
    print(f"period range   {df['period_end'].min().date()} -> {df['period_end'].max().date()}")
    print("\nfield coverage (non-null share):")
    for c in ["revenue", "pat", "net_worth", "total_debt", "shares_outstanding", "cfo"]:
        if c in df:
            print(f"  {c:20} {df[c].notna().mean():6.1%}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
