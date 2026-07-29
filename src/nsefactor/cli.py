"""Command-line entry point: ``nsefactor <command>``."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config import DATA_DIR, DEFAULT_CONFIG


def _fetch(args: argparse.Namespace) -> int:
    from . import bhavcopy as bc

    panel = bc.load_range(args.start, args.end, strict=not args.allow_gaps)
    if panel.empty:
        logging.error("no data fetched")
        return 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "bhavcopy.parquet"
    panel.to_parquet(out, index=False)
    days = bc.trading_days(panel)
    print(f"wrote {out}: {len(panel):,} rows, {len(days):,} days, {days[0].date()} -> {days[-1].date()}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(DATA_DIR.parents[1] / "scripts"))
    import validate  # type: ignore[import-not-found]

    validate.main()
    return 0


def _universe(args: argparse.Namespace) -> int:
    from . import universe as uni

    panel = pd.read_parquet(DATA_DIR / "bhavcopy.parquet")
    as_of = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    sel = uni.select(panel, as_of)
    print(f"universe as of {as_of.date()}: {len(sel)} names\n")
    print(sel[["symbol", "median_turnover", "last_close"]].head(args.top).to_string())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nsefactor")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download bhavcopy history")
    p.add_argument("--start", default=DEFAULT_CONFIG.start_date)
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument(
        "--allow-gaps",
        action="store_true",
        help="do not fail when days cannot be fetched (leaves holes in the panel)",
    )
    p.set_defaults(func=_fetch)

    p = sub.add_parser("validate", help="data-quality report")
    p.set_defaults(func=_validate)

    p = sub.add_parser("universe", help="show the point-in-time universe")
    p.add_argument("--as-of", default=None)
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(func=_universe)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
