"""Invariants for the live surprise monitor."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nsefactor import intraday


class TestMarketHours:
    def test_open_during_session(self):
        # Thursday 11:00 IST
        t = pd.Timestamp("2026-08-06 11:00", tz=intraday.MARKET_TZ)
        assert intraday.market_is_open(t)

    def test_closed_before_open_and_after_close(self):
        for hhmm in ("08:00", "16:00"):
            t = pd.Timestamp(f"2026-08-06 {hhmm}", tz=intraday.MARKET_TZ)
            assert not intraday.market_is_open(t)

    def test_boundaries_are_inclusive(self):
        for hhmm in ("09:15", "15:30"):
            t = pd.Timestamp(f"2026-08-06 {hhmm}", tz=intraday.MARKET_TZ)
            assert intraday.market_is_open(t)

    def test_closed_at_weekend(self):
        sat = pd.Timestamp("2026-08-08 11:00", tz=intraday.MARKET_TZ)
        assert not intraday.market_is_open(sat)

    def test_utc_input_is_converted(self):
        """05:30 UTC is 11:00 IST, mid-session."""
        t = pd.Timestamp("2026-08-06 05:30", tz="UTC")
        assert intraday.market_is_open(t)


class TestExpectedMove:
    def test_includes_the_mad_factor(self):
        """Omitting sqrt(2/pi) is the standard error; pin it down."""
        got = intraday.expected_abs_move(0.252)
        naive = 0.252 / math.sqrt(252)
        assert got == pytest.approx(naive * math.sqrt(2 / math.pi))
        assert got < naive, "expected |move| is below one standard deviation"

    def test_scales_linearly_with_vol(self):
        assert intraday.expected_abs_move(0.40) == pytest.approx(
            2 * intraday.expected_abs_move(0.20))


class TestSurpriseTable:
    def _inputs(self, price_now: float):
        forecasts = [{"symbol": "AAA", "forecast": 25.2, "price": 100.0}]
        idx = pd.date_range("2026-08-06 09:15", periods=3, freq="5min",
                            tz=intraday.MARKET_TZ)
        intra = pd.DataFrame({"AAA": [100.0, 100.5, price_now]}, index=idx)
        prev = pd.Series({"AAA": 100.0})
        return forecasts, intra, prev

    def test_typical_day_scores_about_one(self):
        """A move exactly equal to the expected absolute move is surprise 1.0."""
        expected = intraday.expected_abs_move(0.252)
        forecasts, intra, prev = self._inputs(100.0 * (1 + expected))
        row = intraday.surprise_table(forecasts, intra, prev)[0]
        assert row["surprise"] == pytest.approx(1.0, abs=0.02)

    def test_large_move_scores_high(self):
        expected = intraday.expected_abs_move(0.252)
        forecasts, intra, prev = self._inputs(100.0 * (1 + 3 * expected))
        row = intraday.surprise_table(forecasts, intra, prev)[0]
        assert row["surprise"] == pytest.approx(3.0, abs=0.05)

    def test_direction_does_not_matter(self):
        expected = intraday.expected_abs_move(0.252)
        up = intraday.surprise_table(*self._inputs(100.0 * (1 + 2 * expected)))[0]
        down = intraday.surprise_table(*self._inputs(100.0 * (1 - 2 * expected)))[0]
        assert up["surprise"] == pytest.approx(down["surprise"], abs=0.02)
        assert up["move"] > 0 > down["move"]

    def test_calm_stock_is_more_surprising_for_the_same_move(self):
        """Surprise is relative to each stock's own forecast, not the market."""
        idx = pd.date_range("2026-08-06 09:15", periods=2, freq="5min",
                            tz=intraday.MARKET_TZ)
        intra = pd.DataFrame({"CALM": [100.0, 103.0], "WILD": [100.0, 103.0]}, index=idx)
        prev = pd.Series({"CALM": 100.0, "WILD": 100.0})
        forecasts = [{"symbol": "CALM", "forecast": 12.0},
                     {"symbol": "WILD", "forecast": 60.0}]
        rows = {r["symbol"]: r for r in intraday.surprise_table(forecasts, intra, prev)}
        assert rows["CALM"]["surprise"] > rows["WILD"]["surprise"]

    def test_sorted_most_surprising_first(self):
        idx = pd.date_range("2026-08-06 09:15", periods=2, freq="5min",
                            tz=intraday.MARKET_TZ)
        intra = pd.DataFrame({"A": [100.0, 101.0], "B": [100.0, 108.0]}, index=idx)
        prev = pd.Series({"A": 100.0, "B": 100.0})
        forecasts = [{"symbol": "A", "forecast": 25.0}, {"symbol": "B", "forecast": 25.0}]
        rows = intraday.surprise_table(forecasts, intra, prev)
        assert [r["symbol"] for r in rows] == ["B", "A"]

    def test_missing_symbols_are_skipped_not_guessed(self):
        forecasts = [{"symbol": "AAA", "forecast": 25.0},
                     {"symbol": "GONE", "forecast": 30.0}]
        idx = pd.date_range("2026-08-06 09:15", periods=2, freq="5min",
                            tz=intraday.MARKET_TZ)
        intra = pd.DataFrame({"AAA": [100.0, 101.0]}, index=idx)
        prev = pd.Series({"AAA": 100.0})
        rows = intraday.surprise_table(forecasts, intra, prev)
        assert [r["symbol"] for r in rows] == ["AAA"]

    def test_empty_inputs_return_empty(self):
        assert intraday.surprise_table([], pd.DataFrame(), pd.Series(dtype=float)) == []

    def test_zero_previous_close_is_skipped(self):
        forecasts = [{"symbol": "AAA", "forecast": 25.0}]
        idx = pd.date_range("2026-08-06 09:15", periods=2, freq="5min",
                            tz=intraday.MARKET_TZ)
        intra = pd.DataFrame({"AAA": [100.0, 101.0]}, index=idx)
        assert intraday.surprise_table(forecasts, intra, pd.Series({"AAA": 0.0})) == []
