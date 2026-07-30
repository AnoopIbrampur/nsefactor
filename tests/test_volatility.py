"""Invariants for volatility features and targets.

The failure mode here is a target that bleeds into its own features, which
produces spectacular test-set scores and no forecasting ability whatsoever.
These tests pin down exactly which days each quantity is allowed to see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nsefactor import volatility as vol
from tests.test_data import make_panel


def noisy_panel(n_days: int = 700, n_isins: int = 12, seed: int = 0) -> pd.DataFrame:
    """Panel with real return variation, and deliberately unequal vol levels.

    Equal-volatility instruments would hide the exact bias the log-ratio
    target exists to prevent, so each name gets its own vol scale.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    rows = []
    for i in range(n_isins):
        isin = f"INE000A010{i:02d}"
        scale = 0.008 * (1 + i)  # 0.8% to ~10% daily vol across names
        price = 100.0
        for d in dates:
            r = rng.normal(0, scale)
            price *= 1 + r
            hi = price * (1 + abs(rng.normal(0, scale / 2)))
            lo = price * (1 - abs(rng.normal(0, scale / 2)))
            rows.append(
                {
                    "date": d,
                    "symbol": f"SYM{i}",
                    "isin": isin,
                    "open": price,
                    "high": max(hi, price),
                    "low": min(lo, price),
                    "close": price,
                    "adj_close": price,
                    "prevclose": price / (1 + r),
                    "volume": 1000,
                    "turnover": 1e8,
                    "trades": 100,
                }
            )
    return pd.DataFrame(rows)


class TestForwardTarget:
    def test_forward_vol_uses_exactly_the_next_h_days(self):
        """The target at t must equal the std of returns t+1 .. t+h."""
        panel = noisy_panel(n_days=300, n_isins=3)
        rets = vol.returns_wide(panel)
        h = 21
        fwd = vol.forward_vol(rets, h)

        col = rets.columns[0]
        t_pos = 100
        t = rets.index[t_pos]
        expected = rets[col].iloc[t_pos + 1 : t_pos + 1 + h].std() * np.sqrt(vol.YEAR)
        assert fwd[col].iloc[t_pos] == pytest.approx(expected, rel=1e-9)

    def test_forward_vol_excludes_current_day(self):
        """A huge move *on* date t must not enter the target dated t."""
        panel = noisy_panel(n_days=300, n_isins=3)
        rets = vol.returns_wide(panel)
        h = 21
        base = vol.forward_vol(rets, h)

        col = rets.columns[0]
        t_pos = 100
        spiked = rets.copy()
        spiked.iloc[t_pos, spiked.columns.get_loc(col)] = 0.4

        after = vol.forward_vol(spiked, h)
        assert after[col].iloc[t_pos] == pytest.approx(base[col].iloc[t_pos], rel=1e-9)

    def test_forward_vol_reacts_to_next_day(self):
        """Sanity check the previous test is not vacuous."""
        panel = noisy_panel(n_days=300, n_isins=3)
        rets = vol.returns_wide(panel)
        col = rets.columns[0]
        t_pos = 100
        base = vol.forward_vol(rets, 21)

        spiked = rets.copy()
        spiked.iloc[t_pos + 1, spiked.columns.get_loc(col)] = 0.4
        after = vol.forward_vol(spiked, 21)
        assert after[col].iloc[t_pos] > base[col].iloc[t_pos] * 1.2


class TestFeatureCausality:
    @pytest.mark.parametrize(
        "fn",
        [
            lambda r: vol.realized_vol(r, 21),
            lambda r: vol.realized_vol(r, 126),
            lambda r: vol.ewma_vol(r),
        ],
    )
    def test_truncation_invariance(self, fn):
        """Computing on a truncated history must not change earlier values."""
        panel = noisy_panel(n_days=400, n_isins=4)
        rets = vol.returns_wide(panel)
        cut = 300

        full = fn(rets).iloc[:cut]
        trunc = fn(rets.iloc[:cut])
        pd.testing.assert_frame_equal(full, trunc)

    def test_future_spike_cannot_move_features(self):
        panel = noisy_panel(n_days=400, n_isins=4)
        rets = vol.returns_wide(panel)
        cut = 300
        before = vol.realized_vol(rets, 21).iloc[:cut]

        tampered = rets.copy()
        tampered.iloc[cut + 10 :, 0] *= 20.0
        after = vol.realized_vol(tampered, 21).iloc[:cut]
        pd.testing.assert_frame_equal(before, after)

    def test_parkinson_is_causal(self):
        panel = noisy_panel(n_days=400, n_isins=4)
        cut = 300
        dates = pd.DatetimeIndex(sorted(panel["date"].unique()))

        full = vol.parkinson_vol(panel, 21).loc[: dates[cut - 1]]
        trunc = vol.parkinson_vol(panel[panel["date"] <= dates[cut - 1]], 21)
        pd.testing.assert_frame_equal(full, trunc)


class TestDataset:
    def test_target_is_log_ratio_of_forward_to_anchor(self):
        panel = noisy_panel(n_days=500, n_isins=6)
        df = vol.build_dataset(panel, horizon=21)
        assert len(df) > 0
        recomputed = np.log(df["forward_vol"] / df["anchor"])
        np.testing.assert_allclose(df["target_log_ratio"], recomputed, rtol=1e-12)

    def test_all_feature_columns_present(self):
        panel = noisy_panel(n_days=500, n_isins=6)
        df = vol.build_dataset(panel, horizon=21)
        missing = set(vol.FEATURE_COLUMNS) - set(df.columns)
        assert not missing, f"missing features: {missing}"

    def test_no_infinite_values_in_features(self):
        panel = noisy_panel(n_days=500, n_isins=6)
        df = vol.build_dataset(panel, horizon=21)
        block = df[vol.FEATURE_COLUMNS].to_numpy(dtype=float)
        assert not np.isinf(block).any(), "infinite feature values will break training"

    def test_log_ratio_target_removes_cross_stock_level_bias(self):
        """The reason the target is a ratio at all.

        Names in this fixture span a 12x range of baseline volatility. The
        raw forward vol therefore differs enormously across them, while the
        log ratio should be centred near zero for every one -- which is what
        lets a single model serve all of them.
        """
        panel = noisy_panel(n_days=600, n_isins=12)
        df = vol.build_dataset(panel, horizon=21)

        raw_spread = df.groupby("isin")["forward_vol"].mean()
        ratio_spread = df.groupby("isin")["target_log_ratio"].mean()

        assert raw_spread.max() / raw_spread.min() > 5, "fixture should span vol levels"
        assert ratio_spread.abs().max() < 0.3, (
            "log-ratio target should be near zero for every stock regardless "
            f"of its vol level, got {ratio_spread.abs().max():.3f}"
        )

    def test_universe_filter_restricts_rows(self):
        panel = noisy_panel(n_days=400, n_isins=6)
        keep = pd.Index(["INE000A01000", "INE000A01001"])
        df = vol.build_dataset(panel, horizon=21, universe_isins=keep)
        assert set(df["isin"].unique()) <= set(keep)


class TestBaselines:
    def test_ewma_baseline_is_the_anchor(self):
        panel = noisy_panel(n_days=500, n_isins=6)
        df = vol.build_dataset(panel, horizon=21)
        pd.testing.assert_series_equal(
            vol.baseline_ewma(df), df["anchor"], check_names=False
        )

    def test_persistence_baseline_is_trailing_rv(self):
        panel = noisy_panel(n_days=500, n_isins=6)
        df = vol.build_dataset(panel, horizon=21)
        pd.testing.assert_series_equal(
            vol.baseline_persistence(df), df["rv_21"], check_names=False
        )

    def test_baselines_are_in_a_plausible_range(self):
        """Annualised vol for these fixtures should be double digits, not 0.001."""
        panel = noisy_panel(n_days=600, n_isins=8)
        df = vol.build_dataset(panel, horizon=21)
        for fn in (vol.baseline_ewma, vol.baseline_persistence):
            f = fn(df).dropna()
            assert f.median() > 0.05, "annualised vol implausibly low"
            assert f.median() < 5.0, "annualised vol implausibly high"
