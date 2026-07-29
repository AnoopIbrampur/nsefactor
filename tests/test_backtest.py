"""Invariants for factors, timing, and cost accounting.

A backtest bug never raises -- it just makes the equity curve nicer. These
target the specific ways that happens: signals that peek, entries that
transact before they could have, and delisted names that quietly vanish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nsefactor import backtest, factors, metrics
from nsefactor.config import Config
from tests.test_data import make_panel


def with_adj(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach adj_close without invoking corporate-action detection."""
    out = panel.copy()
    out["adj_close"] = out["close"]
    return out


class TestFactorCausality:
    """No factor may read a bar dated after its formation date."""

    @pytest.mark.parametrize(
        "fn",
        [
            lambda p, d: factors.momentum(p, d, 231),
            lambda p, d: factors.volatility(p, d, 126),
            lambda p, d: factors.reversal(p, d, 21),
            lambda p, d: factors.illiquidity(p, d, 126),
        ],
    )
    def test_truncation_invariance(self, fn):
        panel = with_adj(make_panel(n_days=500))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        as_of = days[400]

        full = fn(panel, as_of)
        truncated = fn(panel[panel["date"] <= as_of], as_of)
        pd.testing.assert_series_equal(full, truncated)

    def test_composite_truncation_invariance(self):
        panel = with_adj(make_panel(n_days=500))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        as_of = days[400]

        full = factors.compute_all(panel, as_of)
        trunc = factors.compute_all(panel[panel["date"] <= as_of], as_of)
        pd.testing.assert_frame_equal(full, trunc)

    def test_future_spike_cannot_change_score(self):
        """Concretely: 10x a stock after the formation date. Score is unmoved."""
        panel = with_adj(make_panel(n_days=500))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        as_of = days[400]
        before = factors.compute_all(panel, as_of)["composite"]

        tampered = panel.copy()
        future = (tampered["date"] > as_of) & (tampered["isin"] == "INE000A01001")
        tampered.loc[future, "adj_close"] *= 10.0

        after = factors.compute_all(tampered, as_of)["composite"]
        pd.testing.assert_series_equal(before, after)


class TestMomentumConstruction:
    def test_skip_month_excludes_recent_window(self):
        """A crash inside the skipped month must not enter 12-1 momentum."""
        panel = with_adj(make_panel(n_days=400))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        as_of = days[350]
        base = factors.momentum(panel, as_of, 231, skip=21)

        crashed = panel.copy()
        recent = (crashed["date"] > days[350 - 21]) & (crashed["date"] <= as_of)
        crashed.loc[recent & (crashed["isin"] == "INE000A01001"), "adj_close"] *= 0.5

        after = factors.momentum(crashed, as_of, 231, skip=21)
        assert after["INE000A01001"] == pytest.approx(base["INE000A01001"])

    def test_crash_before_skip_window_does_register(self):
        """Sanity check the previous test is not passing vacuously."""
        panel = with_adj(make_panel(n_days=400))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        as_of = days[350]
        base = factors.momentum(panel, as_of, 231, skip=21)

        crashed = panel.copy()
        early = crashed["date"] <= days[350 - 100]
        crashed.loc[early & (crashed["isin"] == "INE000A01001"), "adj_close"] *= 0.5

        after = factors.momentum(crashed, as_of, 231, skip=21)
        assert after["INE000A01001"] != pytest.approx(base["INE000A01001"])


class TestForwardReturns:
    def test_simple_return(self):
        panel = with_adj(make_panel(n_days=100))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        isins = pd.Index(["INE000A01001"])
        r = backtest.forward_returns(panel, days[10], days[30], isins)

        wide = panel[panel["isin"] == "INE000A01001"].set_index("date")["adj_close"]
        assert r["INE000A01001"] == pytest.approx(wide[days[30]] / wide[days[10]] - 1)

    def test_delisted_name_exits_at_last_price_not_dropped(self):
        """The survivorship trap: a name that stops trading must book its loss."""
        panel = with_adj(make_panel(n_days=100))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        target = "INE000A01001"

        crashed = panel.copy()
        # Collapse to 10% of value, then delist entirely partway through.
        mid = days[20]
        crashed.loc[(crashed["isin"] == target) & (crashed["date"] == mid), "adj_close"] *= 0.1
        crashed = crashed[~((crashed["isin"] == target) & (crashed["date"] > mid))]

        r = backtest.forward_returns(crashed, days[10], days[30], pd.Index([target]))
        assert r[target] < -0.8, "delisting loss must be realised, not dropped"
        assert not np.isnan(r[target])

    def test_missing_at_entry_is_nan(self):
        """Not buyable at entry means excluded, not assumed free."""
        panel = with_adj(make_panel(n_days=100))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        target = "INE000A01001"
        gone = panel[~((panel["isin"] == target) & (panel["date"] == days[10]))]

        r = backtest.forward_returns(gone, days[10], days[30], pd.Index([target]))
        assert np.isnan(r[target])


class TestTiming:
    def test_entry_is_after_formation(self):
        """Signals form at t's close; the book cannot be entered until t+1."""
        panel = with_adj(make_panel(n_days=500))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        res = backtest.run(panel, lambda p, d: pd.Series(1.0, index=["INE000A01001"]), days)

        assert not res["ledger"].empty
        assert (res["ledger"]["entry"] > res["ledger"]["form_date"]).all()

    def test_periods_do_not_overlap(self):
        panel = with_adj(make_panel(n_days=500))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        res = backtest.run(panel, lambda p, d: pd.Series(1.0, index=["INE000A01001"]), days)
        led = res["ledger"]
        assert (led["entry"].iloc[1:].to_numpy() > led["exit"].iloc[:-1].to_numpy()).all()


class TestCosts:
    def _cfg(self, **kw):
        return Config(**{**Config().__dict__, **kw})

    def test_full_rotation_costs_both_sides(self):
        """Replacing the whole book pays the per-side rate twice."""
        panel = with_adj(make_panel(n_days=500))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        cfg = self._cfg(n_holdings=1, cost_bps_per_side=35.0)

        flip = {"i": 0}

        def alternating(p, d):
            flip["i"] += 1
            name = "INE000A01001" if flip["i"] % 2 else "INE000A01002"
            return pd.Series(1.0, index=[name])

        led = backtest.run(panel, alternating, days, cfg)["ledger"]
        steady = led.iloc[1:]
        assert (steady["turnover"] - 1.0).abs().max() < 1e-9
        assert (steady["cost"] - 2 * 35.0 / 1e4).abs().max() < 1e-9

    def test_no_rotation_costs_nothing(self):
        panel = with_adj(make_panel(n_days=500))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        cfg = self._cfg(n_holdings=1)

        led = backtest.run(panel, lambda p, d: pd.Series(1.0, index=["INE000A01001"]), days, cfg)["ledger"]
        assert led["cost"].iloc[1:].abs().max() < 1e-12

    def test_net_is_gross_minus_cost(self):
        panel = with_adj(make_panel(n_days=500))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        led = backtest.run(panel, lambda p, d: pd.Series(1.0, index=["INE000A01001"]), days)["ledger"]
        assert np.allclose(led["net_return"], led["gross_return"] - led["cost"])

    def test_costs_reduce_returns(self):
        panel = with_adj(make_panel(n_days=500))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        cheap = backtest.run(panel, lambda p, d: pd.Series(1.0, index=["INE000A01001"]), days, self._cfg(cost_bps_per_side=0.0))
        dear = backtest.run(panel, lambda p, d: pd.Series(1.0, index=["INE000A01001"]), days, self._cfg(cost_bps_per_side=100.0))
        assert dear["ledger"]["net_return"].sum() <= cheap["ledger"]["net_return"].sum()


class TestBuffer:
    """Rank buffering must cut turnover without silently changing the book."""

    def _scores(self, order):
        return pd.Series(np.linspace(1.0, 0.0, len(order)), index=order)

    def test_unbuffered_takes_strict_top_n(self):
        names = [f"S{i}" for i in range(10)]
        w = backtest._weights_from_scores(self._scores(names), 3)
        assert list(w.index) == ["S0", "S1", "S2"]
        assert w.sum() == pytest.approx(1.0)

    def test_buffer_retains_held_name_inside_band(self):
        """A held name that slipped to rank 4 of a top-3 book is kept at 2x."""
        names = [f"S{i}" for i in range(10)]
        held = pd.Index(["S3"])
        w = backtest._weights_from_scores(self._scores(names), 3, held=held, buffer_mult=2.0)
        assert "S3" in w.index, "held name inside the buffer band should be retained"
        assert len(w) == 3

    def test_buffer_sells_name_outside_band(self):
        """Slipping past the wider band forces the sale."""
        names = [f"S{i}" for i in range(10)]
        held = pd.Index(["S8"])  # rank 9, outside top 6
        w = backtest._weights_from_scores(self._scores(names), 3, held=held, buffer_mult=2.0)
        assert "S8" not in w.index
        assert list(w.index) == ["S0", "S1", "S2"]

    def test_buffer_never_exceeds_n_holdings(self):
        names = [f"S{i}" for i in range(20)]
        held = pd.Index(["S3", "S4", "S5"])
        w = backtest._weights_from_scores(self._scores(names), 3, held=held, buffer_mult=3.0)
        assert len(w) == 3
        assert w.sum() == pytest.approx(1.0)

    def test_buffer_reduces_turnover_on_real_run(self):
        panel = with_adj(make_panel(n_days=600, isins=tuple(f"INE000A0100{i}" for i in range(9))))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        cfg = Config(**{**Config().__dict__, "n_holdings": 3})

        rng = np.random.default_rng(0)
        isins = sorted(panel["isin"].unique())

        def noisy(p, as_of):
            # Scores that jitter around a stable ordering: exactly the case
            # buffering is meant to help.
            base = np.linspace(1.0, 0.0, len(isins))
            return pd.Series(base + rng.normal(0, 0.15, len(isins)), index=isins)

        plain = backtest.run(panel, noisy, days, cfg, buffer_mult=1.0)["ledger"]
        rng = np.random.default_rng(0)
        buffed = backtest.run(panel, noisy, days, cfg, buffer_mult=2.0)["ledger"]

        assert buffed["turnover"].mean() < plain["turnover"].mean()


class TestFactorSelection:
    def test_use_restricts_composite(self):
        # zscore needs >=10 names in the cross-section, or every composite is
        # NaN and the comparison passes vacuously.
        panel = with_adj(make_panel(n_days=500, isins=tuple(f"INE000A010{i:02d}" for i in range(15))))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        as_of = days[400]

        full = factors.compute_all(panel, as_of)
        subset = factors.compute_all(panel, as_of, use=("mom_12_1", "vol_126"))

        # All z-scores are still reported either way; only the blend changes.
        assert "z_illiq_126" in subset.columns
        assert not full["composite"].equals(subset["composite"])
        expected = subset[["z_mom_12_1", "z_vol_126"]].mean(axis=1, skipna=True)
        pd.testing.assert_series_equal(
            subset["composite"].dropna(), expected.dropna(), check_names=False
        )

    def test_unknown_factor_raises(self):
        panel = with_adj(make_panel(n_days=500))
        days = pd.DatetimeIndex(sorted(panel["date"].unique()))
        with pytest.raises(KeyError):
            factors.compute_all(panel, days[400], use=("nonexistent",))


class TestMetrics:
    def test_cagr_of_known_series(self):
        r = pd.Series([0.01] * 12)
        assert metrics.cagr(r) == pytest.approx(1.01**12 - 1)

    def test_max_drawdown(self):
        r = pd.Series([0.5, -0.5, 0.0])
        # 1.5 then 0.75 -> peak-to-trough of -50%
        assert metrics.max_drawdown(r) == pytest.approx(-0.5)

    def test_zero_vol_sharpe_is_nan(self):
        assert np.isnan(metrics.sharpe(pd.Series([0.01] * 10)))
