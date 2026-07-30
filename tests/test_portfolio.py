"""Invariants for risk-aware portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nsefactor import portfolio


class TestInverseVolWeights:
    def test_weights_sum_to_one(self):
        chosen = pd.Index([f"S{i}" for i in range(10)])
        risk = pd.Series(np.linspace(0.15, 0.60, 10), index=chosen)
        w = portfolio.inverse_vol_weights(chosen, risk)
        assert w.sum() == pytest.approx(1.0)

    def test_calmer_stock_gets_more_weight(self):
        chosen = pd.Index(["calm", "jumpy"])
        risk = pd.Series({"calm": 0.15, "jumpy": 0.60})
        w = portfolio.inverse_vol_weights(chosen, risk, max_weight=1.0)
        assert w["calm"] > w["jumpy"]
        # 1/0.15 vs 1/0.60 is a 4:1 ratio
        assert w["calm"] / w["jumpy"] == pytest.approx(4.0, rel=1e-6)

    def test_max_weight_respected(self):
        chosen = pd.Index([f"S{i}" for i in range(20)])
        # One name far calmer than the rest would otherwise dominate.
        risk = pd.Series([0.02] + [0.50] * 19, index=chosen)
        w = portfolio.inverse_vol_weights(chosen, risk, max_weight=0.08)
        assert w.max() <= 0.08 + 1e-9
        assert w.sum() == pytest.approx(1.0)

    def test_cap_below_one_over_n_is_infeasible_and_relaxed(self):
        """A 5% cap across 10 names cannot sum to 1.0; take 1/n instead of looping."""
        chosen = pd.Index([f"S{i}" for i in range(10)])
        risk = pd.Series(np.linspace(0.15, 0.6, 10), index=chosen)
        w = portfolio.inverse_vol_weights(chosen, risk, max_weight=0.05)
        assert w.sum() == pytest.approx(1.0)
        assert np.allclose(w, 0.1), "infeasible cap should collapse to equal weight"

    def test_many_calm_names_all_capped(self):
        """Every name at the cap must still sum to one, not to n*cap."""
        chosen = pd.Index([f"S{i}" for i in range(10)])
        risk = pd.Series([0.2] * 10, index=chosen)
        w = portfolio.inverse_vol_weights(chosen, risk, max_weight=0.10)
        assert w.sum() == pytest.approx(1.0)
        assert w.max() <= 0.10 + 1e-9

    def test_floor_vol_prevents_riskless_sizing(self):
        chosen = pd.Index(["a", "b"])
        risk = pd.Series({"a": 1e-6, "b": 0.30})
        w = portfolio.inverse_vol_weights(chosen, risk, max_weight=1.0, floor_vol=0.10)
        # Without the floor, 'a' would take essentially the whole book.
        assert w["a"] / w["b"] == pytest.approx(3.0, rel=1e-6)

    def test_missing_risk_falls_back_to_median(self):
        chosen = pd.Index(["a", "b", "c"])
        risk = pd.Series({"a": 0.20, "b": 0.40})  # 'c' missing
        w = portfolio.inverse_vol_weights(chosen, risk, max_weight=1.0)
        assert w.notna().all()
        assert w.sum() == pytest.approx(1.0)

    def test_all_missing_risk_gives_equal_weight(self):
        chosen = pd.Index(["a", "b", "c"])
        w = portfolio.inverse_vol_weights(chosen, pd.Series(dtype=float), max_weight=1.0)
        assert np.allclose(w, 1 / 3)

    def test_reduces_portfolio_volatility_vs_equal_weight(self):
        """The whole point: risk sizing should lower realised portfolio vol."""
        rng = np.random.default_rng(0)
        n, t = 20, 2000
        vols = np.linspace(0.10, 0.80, n)
        rets = rng.normal(0, vols / np.sqrt(252), size=(t, n))
        chosen = pd.Index([f"S{i}" for i in range(n)])
        risk = pd.Series(vols, index=chosen)

        w_iv = portfolio.inverse_vol_weights(chosen, risk, max_weight=0.30).to_numpy()
        w_eq = np.full(n, 1 / n)

        assert (rets @ w_iv).std() < (rets @ w_eq).std()


class TestGroupCap:
    # Four groups, so a 30% cap is actually satisfiable (4 x 0.30 = 1.2 >= 1.0).
    # With three groups it is not, and the function correctly relaxes instead.
    W = pd.Series({"a": 0.40, "b": 0.30, "c": 0.15, "d": 0.10, "e": 0.05})
    G = pd.Series({"a": "metals", "b": "metals", "c": "it", "d": "pharma", "e": "banks"})

    def test_breached_group_is_capped(self):
        out = portfolio.cap_by_group(self.W, self.G, max_group=0.30)
        assert out.groupby(self.G).sum()["metals"] <= 0.30 + 1e-9
        assert out.sum() == pytest.approx(1.0)

    def test_unbreached_groups_untouched(self):
        w = pd.Series({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
        g = pd.Series({"a": "x", "b": "y", "c": "z", "d": "w"})
        out = portfolio.cap_by_group(w, g, max_group=0.30)
        pd.testing.assert_series_equal(out, w, check_names=False)

    def test_no_group_exceeds_cap_after_cascade(self):
        """Capping one group must not re-inflate a group already capped.

        The naive loop capped metals, spilled into IT, capped IT, spilled into
        pharma, then capped pharma and spilled straight back into metals.
        """
        out = portfolio.cap_by_group(self.W, self.G, max_group=0.30)
        totals = out.groupby(self.G).sum()
        assert (totals <= 0.30 + 1e-9).all(), f"cap breached: {dict(totals.round(4))}"
        assert out.sum() == pytest.approx(1.0)

    def test_infeasible_group_cap_relaxes_to_one_over_groups(self):
        """Two groups cannot both stay under 30% of a book summing to 100%."""
        w = pd.Series({"a": 0.6, "b": 0.4})
        g = pd.Series({"a": "x", "b": "y"})
        out = portfolio.cap_by_group(w, g, max_group=0.30)
        assert out.sum() == pytest.approx(1.0)
        assert (out.groupby(g).sum() <= 0.5 + 1e-9).all()


class TestSelectWithBuffer:
    def test_strict_top_n_without_buffer(self):
        s = pd.Series(np.linspace(1, 0, 10), index=[f"S{i}" for i in range(10)])
        assert list(portfolio.select_with_buffer(s, 3, buffer_mult=1.0)) == ["S0", "S1", "S2"]

    def test_held_name_inside_band_retained(self):
        s = pd.Series(np.linspace(1, 0, 10), index=[f"S{i}" for i in range(10)])
        got = portfolio.select_with_buffer(s, 3, held=pd.Index(["S4"]), buffer_mult=2.0)
        assert "S4" in got
        assert len(got) == 3

    def test_held_name_outside_band_dropped(self):
        s = pd.Series(np.linspace(1, 0, 10), index=[f"S{i}" for i in range(10)])
        got = portfolio.select_with_buffer(s, 3, held=pd.Index(["S9"]), buffer_mult=2.0)
        assert "S9" not in got


class TestBuild:
    def test_build_returns_normalised_weights(self):
        s = pd.Series(np.linspace(1, 0, 30), index=[f"S{i}" for i in range(30)])
        risk = pd.Series(np.linspace(0.15, 0.7, 30), index=s.index)
        w = portfolio.build(s, risk, n_holdings=10, max_weight=0.20)
        assert len(w) == 10
        assert w.sum() == pytest.approx(1.0)
        assert w.max() <= 0.20 + 1e-9

    def test_build_without_risk_is_equal_weight(self):
        s = pd.Series(np.linspace(1, 0, 30), index=[f"S{i}" for i in range(30)])
        w = portfolio.build(s, None, n_holdings=10)
        assert np.allclose(w, 0.1)

    def test_empty_scores_yields_empty_book(self):
        w = portfolio.build(pd.Series(dtype=float), None, n_holdings=10)
        assert w.empty
