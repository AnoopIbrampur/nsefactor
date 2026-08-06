"""Live monitoring against the forecast, for the intraday refresh.

Re-running the model intraday would be pointless: it forecasts volatility over
the next 21 trading days from daily bars, so recomputing it every fifteen
minutes returns the same number. What *is* worth watching live is whether
today is behaving the way the forecast said it would.

That inverts the model's known weakness into the thing this view is for. The
forecast is good at anticipating volatility *settling* and poor at
anticipating it *spiking* -- so it cannot warn about a jump, but a live monitor
can detect one while it is happening.

Surprise metric
---------------
Under a forecast of ``sigma`` annualised, a single session's expected absolute
return is::

    E|r| = sigma / sqrt(252) * sqrt(2/pi)

The ``sqrt(2/pi)`` factor (~0.798) is the mean absolute deviation of a normal
variable, and leaving it out is the usual mistake: comparing today's move
directly against ``sigma/sqrt(252)`` makes an ordinary day look like a 0.8x
day, and every threshold drawn on top of that is off by a fifth.

So ``surprise = |today's move| / E|r|``, where 1.0 is a textbook-typical
session. Values above ~2.5 are worth a look; a normal distribution puts
roughly 5% of days above 2.24.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252
# Mean absolute deviation of a standard normal: E|Z| = sqrt(2/pi).
MAD_NORMAL = math.sqrt(2.0 / math.pi)

# NSE trades 09:15-15:30 IST.
MARKET_TZ = "Asia/Kolkata"
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)


def market_is_open(now: pd.Timestamp | None = None) -> bool:
    """Whether NSE is in its continuous session right now.

    Weekend-aware but not holiday-aware -- the holiday calendar lives in the
    data itself (a missing bhavcopy is the signal), and hardcoding one here
    would be a second source of truth that silently drifts.
    """
    now = (now or pd.Timestamp.now(tz=MARKET_TZ)).tz_convert(MARKET_TZ)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    close_t = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
    return open_t <= now <= close_t


def fetch_intraday(symbols: list[str], interval: str = "5m") -> pd.DataFrame:
    """Today's intraday bars for NSE symbols, via Yahoo's ``.NS`` tickers.

    Returns a frame indexed by timestamp with one column per plain symbol
    (the ``.NS`` suffix is stripped again on the way out). Empty if the fetch
    fails entirely, which the caller should treat as "no live view", not as
    an error worth failing the whole refresh over.
    """
    import warnings

    import yfinance as yf

    tickers = [f"{s}.NS" for s in symbols]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            raw = yf.download(
                tickers, period="1d", interval=interval,
                progress=False, auto_adjust=False, threads=True,
            )
        except Exception as exc:
            log.warning("intraday fetch failed: %s", exc)
            return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.columns = [c[:-3] if str(c).endswith(".NS") else str(c) for c in close.columns]
    return close.dropna(how="all")


def previous_closes(symbols: list[str]) -> pd.Series:
    """Prior session's close per symbol.

    Taken from the same feed as the intraday bars rather than from our own
    panel: the panel can be a session or two behind between daily refreshes,
    and pairing today's price with a stale reference would manufacture a move
    that never happened.
    """
    import warnings

    import yfinance as yf

    tickers = [f"{s}.NS" for s in symbols]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            hist = yf.download(
                tickers, period="5d", interval="1d",
                progress=False, auto_adjust=False, threads=True,
            )
        except Exception as exc:
            log.warning("previous-close fetch failed: %s", exc)
            return pd.Series(dtype=float)

    if hist.empty:
        return pd.Series(dtype=float)

    close = hist["Close"] if isinstance(hist.columns, pd.MultiIndex) else hist[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.columns = [c[:-3] if str(c).endswith(".NS") else str(c) for c in close.columns]

    # The last row may be today's partial session, which is not a prior close.
    today = pd.Timestamp.now(tz=MARKET_TZ).normalize().date()
    rows = close[[d.date() != today for d in close.index]]
    if rows.empty:
        rows = close.iloc[:-1] if len(close) > 1 else close
    return rows.ffill().iloc[-1]


def expected_abs_move(forecast_vol: float) -> float:
    """Expected absolute one-day return implied by an annualised forecast."""
    return forecast_vol / math.sqrt(TRADING_DAYS) * MAD_NORMAL


def surprise_table(
    forecasts: list[dict],
    intraday: pd.DataFrame,
    prev_close: pd.Series,
) -> list[dict]:
    """Per-stock comparison of today's move against what the forecast implied.

    ``surprise`` is today's absolute move divided by the expected absolute
    move, so 1.0 is a typical session for that stock and larger means it is
    running hot relative to its own forecast -- not relative to the market.
    """
    if intraday.empty or prev_close.empty:
        return []

    latest = intraday.ffill().iloc[-1]
    out = []
    for f in forecasts:
        sym = f["symbol"]
        if sym not in latest.index or sym not in prev_close.index:
            continue
        now_px, prev_px = latest.get(sym), prev_close.get(sym)
        if pd.isna(now_px) or pd.isna(prev_px) or prev_px <= 0:
            continue

        vol = f["forecast"] / 100.0
        expected = expected_abs_move(vol)
        if expected <= 0:
            continue

        move = float(now_px) / float(prev_px) - 1.0
        out.append(
            {
                "symbol": sym,
                "price": round(float(now_px), 2),
                "prev_close": round(float(prev_px), 2),
                "move": round(move * 100, 2),
                "forecast": f["forecast"],
                "expected_move": round(expected * 100, 2),
                "surprise": round(abs(move) / expected, 2),
            }
        )
    out.sort(key=lambda r: r["surprise"], reverse=True)
    return out
