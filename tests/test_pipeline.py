"""Invariants for the scheduled-update path.

The failures that matter here are silent ones: a duplicated session, or a page
that publishes month-old numbers looking exactly like today's.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nsefactor import bhavcopy, pipeline
from tests.test_data import make_panel


@pytest.fixture
def panel_on_disk(tmp_path, monkeypatch):
    """A stored panel ending 2020-06-30, with the module pointed at it."""
    path = tmp_path / "bhavcopy.parquet"
    monkeypatch.setattr(pipeline, "PANEL_PATH", path)
    panel = make_panel(n_days=120, start="2020-01-01")
    panel.to_parquet(path, index=False)
    return path, panel


class TestFreshnessGuard:
    """Stale data must fail loudly rather than publish quietly."""

    def test_recent_data_passes(self):
        now = pd.Timestamp("2026-08-03")
        pipeline.assert_fresh(pd.Timestamp("2026-07-31"), now=now)

    def test_weekend_plus_holiday_does_not_trip_it(self):
        """Friday data read on the following Wednesday is normal, not stale."""
        now = pd.Timestamp("2026-08-05")
        pipeline.assert_fresh(pd.Timestamp("2026-07-31"), now=now, max_age_days=5)

    def test_month_old_data_raises(self):
        now = pd.Timestamp("2026-08-03")
        with pytest.raises(pipeline.StaleDataError, match="Refusing to publish"):
            pipeline.assert_fresh(pd.Timestamp("2026-07-01"), now=now)

    def test_error_names_the_actual_age(self):
        now = pd.Timestamp("2026-08-03")
        with pytest.raises(pipeline.StaleDataError) as exc:
            pipeline.assert_fresh(pd.Timestamp("2026-06-03"), now=now)
        assert "61 days old" in str(exc.value)


class TestIncrementalAppend:
    def test_no_new_sessions_is_a_noop(self, panel_on_disk, monkeypatch):
        path, panel = panel_on_disk
        monkeypatch.setattr(
            bhavcopy, "load_range",
            lambda *a, **k: pd.DataFrame(columns=bhavcopy.COLUMNS))

        before = pd.read_parquet(path)
        result = pipeline.append_new_sessions()
        after = pd.read_parquet(path)

        assert result.sessions_added == 0
        pd.testing.assert_frame_equal(before, after)

    def test_new_sessions_are_appended(self, panel_on_disk, monkeypatch):
        path, panel = panel_on_disk
        last = panel["date"].max()
        extra = make_panel(n_days=3, start=str((last + pd.Timedelta(days=1)).date()))
        monkeypatch.setattr(bhavcopy, "load_range", lambda *a, **k: extra)

        result = pipeline.append_new_sessions()
        stored = pd.read_parquet(path)

        assert result.sessions_added == 3
        assert stored["date"].max() == extra["date"].max()
        assert len(stored) == len(panel) + len(extra)

    def test_refetched_days_do_not_duplicate(self, panel_on_disk, monkeypatch):
        """Re-fetching a stored session must not double its rows."""
        path, panel = panel_on_disk
        overlap = panel[panel["date"] >= panel["date"].max() - pd.Timedelta(days=10)]
        monkeypatch.setattr(bhavcopy, "load_range", lambda *a, **k: overlap)

        pipeline.append_new_sessions()
        stored = pd.read_parquet(path)

        assert len(stored) == len(panel)
        assert not stored.duplicated(subset=["date", "isin"]).any()

    def test_a_corrected_bar_replaces_the_original(self, panel_on_disk, monkeypatch):
        """A restated session should overwrite, not sit alongside."""
        path, panel = panel_on_disk
        last = panel["date"].max()
        fixed = panel[panel["date"] == last].copy()
        fixed["close"] = 999.0
        monkeypatch.setattr(bhavcopy, "load_range", lambda *a, **k: fixed)

        pipeline.append_new_sessions(start_hint=str(last.date()))
        stored = pd.read_parquet(path)
        row = stored[(stored["date"] == last)]

        assert len(row) == len(fixed)
        assert (row["close"] == 999.0).all()

    def test_result_reports_panel_state(self, panel_on_disk, monkeypatch):
        path, panel = panel_on_disk
        monkeypatch.setattr(
            bhavcopy, "load_range",
            lambda *a, **k: pd.DataFrame(columns=bhavcopy.COLUMNS))
        result = pipeline.append_new_sessions()
        assert result.total_sessions == panel["date"].nunique()
        assert result.rows == len(panel)


class TestModelPersistence:
    def test_roundtrip(self, tmp_path):
        from sklearn.ensemble import HistGradientBoostingRegressor
        import numpy as np

        path = tmp_path / "m.joblib"
        m = HistGradientBoostingRegressor(max_iter=5)
        m.fit(np.random.rand(60, 3), np.random.rand(60))
        pipeline.save_model(m, path, {"trained_on": "2026-08-01", "iters": 5})

        loaded, meta = pipeline.load_model(path)
        assert loaded is not None
        assert meta["iters"] == 5
        assert loaded.predict(np.random.rand(4, 3)).shape == (4,)

    def test_missing_model_returns_none(self, tmp_path):
        model, meta = pipeline.load_model(tmp_path / "absent.joblib")
        assert model is None and meta == {}

    def test_model_age(self):
        age = pipeline.model_age_days(
            {"trained_on": "2026-07-01"}, now=pd.Timestamp("2026-08-03"))
        assert age == 33

    def test_model_age_unknown_when_unstamped(self):
        assert pipeline.model_age_days({}) is None
