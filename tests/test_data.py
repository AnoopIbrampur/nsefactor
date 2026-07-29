"""Invariants for the data layer.

The expensive failures in a backtest are silent: a lookahead leak or an
unadjusted split does not raise, it just produces a better-looking equity
curve. These tests target exactly those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nsefactor import adjust, bhavcopy, universe
from nsefactor.config import Config


def make_panel(
    n_days: int = 260,
    isins: tuple[str, ...] = ("INE000A01001", "INE000A01002", "INE000A01003"),
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """A clean synthetic panel: no actions, no gaps, known turnover ordering."""
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for i, isin in enumerate(isins):
        price = 100.0 * (i + 1)
        for d in dates:
            price *= 1.0 + 0.0004 * (i + 1)
            rows.append(
                {
                    "date": d,
                    "symbol": f"SYM{i}",
                    "isin": isin,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "prevclose": price / (1.0 + 0.0004 * (i + 1)),
                    "volume": 1000,
                    # turnover strictly decreasing in i, so rank order is known
                    "turnover": 1e9 / (i + 1),
                    "trades": 100,
                }
            )
    return pd.DataFrame(rows)


def apply_split(panel: pd.DataFrame, isin: str, ex_date: pd.Timestamp, ratio: float) -> pd.DataFrame:
    """Rewrite a panel so ``isin`` undergoes a split on ``ex_date``.

    Mimics NSE's own convention: prices from the ex-date onward are quoted
    post-split, and the ex-date's ``prevclose`` is the adjusted prior close.
    """
    df = panel.copy()
    mask = (df["isin"] == isin) & (df["date"] >= ex_date)
    for col in ("open", "high", "low", "close", "prevclose"):
        df.loc[mask, col] = df.loc[mask, col] * ratio
    # The ex-date prevclose is the *adjusted* previous close; every other
    # session's prevclose stays consistent with its own era.
    ex_row = (df["isin"] == isin) & (df["date"] == ex_date)
    prior = df[(df["isin"] == isin) & (df["date"] < ex_date)]["close"].iloc[-1]
    df.loc[ex_row, "prevclose"] = prior * ratio
    return df


class TestCorporateActions:
    def test_clean_panel_has_no_actions(self):
        actions = adjust.detect_factors(make_panel())
        assert actions.empty, "no corporate actions should be inferred from a clean panel"

    @pytest.mark.parametrize("ratio", [0.5, 0.2, 0.1])
    def test_split_factor_recovered(self, ratio):
        ex = pd.Timestamp("2020-06-01")
        panel = apply_split(make_panel(), "INE000A01001", ex, ratio)
        actions = adjust.detect_factors(panel)

        assert len(actions) == 1
        row = actions.iloc[0]
        assert row["isin"] == "INE000A01001"
        assert row["date"] == ex
        assert row["factor"] == pytest.approx(ratio, rel=1e-6)

    @pytest.mark.parametrize("ratio", [0.5, 0.2, 0.1])
    def test_split_produces_no_phantom_crash(self, ratio):
        """The whole point: an unadjusted 1:10 split reads as a -90% day."""
        ex = pd.Timestamp("2020-06-01")
        panel = apply_split(make_panel(), "INE000A01001", ex, ratio)

        raw = panel[panel["isin"] == "INE000A01001"].sort_values("date")
        raw_ret = raw["close"].pct_change()
        assert raw_ret.min() < -0.4, "test fixture should contain a phantom crash"

        adj = adjust.adjusted_returns(adjust.adjusted_close(panel))
        one = adj[adj["isin"] == "INE000A01001"]
        assert one["ret"].abs().max() < 0.01, "adjustment should remove the phantom move"

    def test_adjustment_preserves_latest_price(self):
        """Back-adjustment must leave today's quoted price recognisable."""
        ex = pd.Timestamp("2020-06-01")
        panel = apply_split(make_panel(), "INE000A01001", ex, 0.5)
        adj = adjust.adjusted_close(panel)

        last = adj.sort_values("date").groupby("isin").tail(1)
        assert np.allclose(last["adj_close"], last["close"])

    def test_unaffected_stocks_untouched(self):
        ex = pd.Timestamp("2020-06-01")
        panel = apply_split(make_panel(), "INE000A01001", ex, 0.5)
        adj = adjust.adjusted_close(panel)

        others = adj[adj["isin"] != "INE000A01001"]
        assert np.allclose(others["adj_factor"], 1.0)
        assert np.allclose(others["adj_close"], others["close"])


class TestCalendarCoverage:
    """NSE's trading calendar is not 'weekdays minus holidays'."""

    def test_muhurat_weekend_session_is_requested(self, monkeypatch):
        """Diwali Muhurat sessions trade on Saturdays and Sundays.

        Skipping them leaves a hole that makes the next session's prevclose
        disagree with the previous close in the panel, which then reads as a
        phantom corporate action on the first weekday after Diwali.
        """
        requested: list = []

        def fake_fetch(day, session=None, retries=5):
            requested.append(day)
            return None  # pretend every day is a holiday

        monkeypatch.setattr(bhavcopy, "fetch_raw", fake_fetch)
        # 2020-11-14 was a Saturday Muhurat session (Diwali).
        bhavcopy.load_range("2020-11-09", "2020-11-16", max_workers=1)

        assert pd.Timestamp("2020-11-14").date() in requested, "Saturday Muhurat session skipped"
        assert pd.Timestamp("2020-11-15").date() in requested, "Sunday not requested"

    def test_holiday_404_is_not_a_failure(self, monkeypatch):
        """A 404 means no session that day and must not raise under strict."""
        monkeypatch.setattr(bhavcopy, "fetch_raw", lambda d, session=None, retries=5: None)
        out = bhavcopy.load_range("2020-11-09", "2020-11-16", max_workers=1, strict=True)
        assert out.empty

    def test_throttled_day_raises_under_strict(self, monkeypatch):
        """A 403 is missing data, not a holiday, and must never pass silently."""

        def throttle(day, session=None, retries=5):
            raise bhavcopy.Throttled("403")

        monkeypatch.setattr(bhavcopy, "fetch_raw", throttle)
        with pytest.raises(bhavcopy.Throttled):
            bhavcopy.load_range("2020-11-09", "2020-11-16", max_workers=1, strict=True)


class TestUniverseCausality:
    """No selection may depend on data after its own as-of date."""

    def test_truncation_invariance(self):
        """Selecting on a truncated panel must equal selecting on the full one.

        This is the lookahead test. If any statistic reaches forward, the two
        results diverge.
        """
        panel = make_panel(n_days=400)
        as_of = pd.Timestamp(sorted(panel["date"].unique())[250])

        full = universe.select(panel, as_of)
        truncated = universe.select(panel[panel["date"] <= as_of], as_of)

        pd.testing.assert_frame_equal(full, truncated)

    def test_future_crash_cannot_change_selection(self):
        """Concretely: destroy a stock *after* the as-of date. Nothing moves."""
        panel = make_panel(n_days=400)
        as_of = pd.Timestamp(sorted(panel["date"].unique())[250])
        before = universe.select(panel, as_of)

        sabotaged = panel.copy()
        future = sabotaged["date"] > as_of
        target = sabotaged["isin"] == "INE000A01001"
        sabotaged.loc[future & target, ["close", "turnover"]] = 0.01

        after = universe.select(sabotaged, as_of)
        pd.testing.assert_frame_equal(before, after)

    def test_delisted_stock_still_selectable_before_delisting(self):
        """A name that vanishes later must remain investable while it traded."""
        panel = make_panel(n_days=400)
        dates = sorted(panel["date"].unique())
        delist_at = pd.Timestamp(dates[300])
        alive = panel[~((panel["isin"] == "INE000A01001") & (panel["date"] >= delist_at))]

        early = universe.select(alive, pd.Timestamp(dates[250]))
        late = universe.select(alive, pd.Timestamp(dates[350]))

        assert "INE000A01001" in set(early.index), "should be present while trading"
        assert "INE000A01001" not in set(late.index), "should drop out once delisted"


class TestUniverseFilters:
    def test_ranks_by_liquidity(self):
        panel = make_panel()
        as_of = pd.Timestamp(sorted(panel["date"].unique())[-1])
        sel = universe.select(panel, as_of)
        assert list(sel["symbol"]) == ["SYM0", "SYM1", "SYM2"]

    def test_turnover_floor_excludes_illiquid(self):
        panel = make_panel()
        panel.loc[panel["isin"] == "INE000A01003", "turnover"] = 1e5  # below floor
        as_of = pd.Timestamp(sorted(panel["date"].unique())[-1])

        sel = universe.select(panel, as_of, cfg=Config())
        assert "INE000A01003" not in set(sel.index)

    def test_intermittent_trader_excluded(self):
        """A stock trading only half the window cannot be ranked."""
        panel = make_panel()
        dates = sorted(panel["date"].unique())
        drop = set(dates[-120:-60])
        panel = panel[~((panel["isin"] == "INE000A01002") & (panel["date"].isin(drop)))]

        sel = universe.select(panel, pd.Timestamp(dates[-1]))
        assert "INE000A01002" not in set(sel.index)

    def test_suspended_on_rebalance_date_excluded(self):
        panel = make_panel()
        dates = sorted(panel["date"].unique())
        as_of = pd.Timestamp(dates[-1])
        panel = panel[~((panel["isin"] == "INE000A01001") & (panel["date"] == as_of))]

        sel = universe.select(panel, as_of)
        assert "INE000A01001" not in set(sel.index)
