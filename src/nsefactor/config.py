"""Central configuration for data sourcing, universe, costs, and backtest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_DIR = ARTIFACTS_DIR / "raw"  # untouched bhavcopy zips, one per session
DATA_DIR = ARTIFACTS_DIR / "data"  # normalised parquet
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# NSE serves archives from a host that, unlike www.nseindia.com, does not
# require the cookie handshake. Everything here is a plain GET.
ARCHIVE_HOST = "https://nsearchives.nseindia.com"

# NSE replaced the legacy bhavcopy with the UDiFF format. Both are published
# for early 2024; we prefer UDiFF from this date on and legacy before it.
UDIFF_CUTOVER = "2024-01-01"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass(frozen=True)
class Config:
    # --- Data ---
    start_date: str = "2015-01-01"
    # Only the rolling-settlement equity series. BE/SM/ST are trade-for-trade,
    # illiquid, or SME boards; treating them as investable inflates backtests.
    equity_series: tuple[str, ...] = ("EQ",)
    # ISINs for Indian companies start with INE. ETFs and fund units (INF...)
    # share the EQ series but are not stocks.
    equity_isin_prefix: str = "INE"

    # --- Universe ---
    universe_size: int = 500
    # A stock must trade at least this fraction of the days in the formation
    # window to be rankable, so we never "buy" something that was suspended.
    min_trading_days_frac: float = 0.90
    # Median daily turnover floor, in INR. Filters names a retail portfolio
    # could not actually accumulate without moving the price.
    min_median_turnover: float = 1e7  # Rs 1 crore

    # --- Backtest ---
    rebalance: str = "ME"  # month-end
    n_holdings: int = 20
    # Round-trip cost assumption, in basis points of traded notional:
    # brokerage + exchange fees + STT (0.1% each side on delivery) + impact.
    cost_bps_per_side: float = 35.0

    # --- Corporate actions ---
    # A prevclose/close mismatch beyond this is treated as a corporate action
    # rather than a data error. 2% comfortably exceeds tick-rounding noise.
    ca_detect_tolerance: float = 0.02
    # Adjustment factors outside this range are implausible as real actions
    # and are more likely bad ticks; they get logged and ignored.
    ca_factor_bounds: tuple[float, float] = (0.02, 2.0)

    seed: int = 42


DEFAULT_CONFIG = Config()
