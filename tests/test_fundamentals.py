"""Point-in-time invariants for fundamentals.

A value factor that reads earnings before they were announced produces a
backtest that cannot be reproduced with real money. None of these failures
raise on their own, so they are tested explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nsefactor import fundamentals as F
from nsefactor import fundfactors as FF


def quarters(n: int, start: str = "2020-03-31", freq: str = "QE"):
    return pd.date_range(start, periods=n, freq=freq)


def make_fund(
    isins=("INE000A01001", "INE000A01002"),
    n_quarters: int = 12,
    lag_days: int = 45,
    pat: float = 100.0,
    net_worth: float = 1000.0,
    shares: float = 1_000_000.0,
) -> pd.DataFrame:
    """Synthetic fundamentals with a realistic reporting lag."""
    rows = []
    for i, isin in enumerate(isins):
        for q in quarters(n_quarters):
            rows.append(
                {
                    "isin": isin,
                    "symbol": f"SYM{i}",
                    "period_end": q,
                    "period_start": q - pd.Timedelta(days=89),
                    "broadcast_date": q + pd.Timedelta(days=lag_days),
                    "pat": pat * (i + 1),
                    "revenue": pat * 10 * (i + 1),
                    "net_worth": net_worth * (i + 1),
                    "total_debt": 500.0 * (i + 1),
                    "shares_outstanding": shares,
                    "consolidated": True,
                    "audited": False,
                }
            )
    return pd.DataFrame(rows)


class TestAsOfNoLookahead:
    """The central rule: nothing is visible before it was broadcast."""

    def test_filing_invisible_before_broadcast(self):
        fund = make_fund(n_quarters=4, lag_days=45)
        q_end = quarters(4)[-1]

        # One day before broadcast, the newest filing must not appear.
        day_before = q_end + pd.Timedelta(days=44)
        got = F.as_of(fund, day_before)
        assert q_end not in set(got["period_end"]), "used a filing before it existed"

        on_day = F.as_of(fund, q_end + pd.Timedelta(days=45))
        assert q_end in set(on_day["period_end"]), "filing should be visible once broadcast"

    def test_quarter_end_alone_does_not_make_data_visible(self):
        """Standing on the quarter end itself, that quarter is not yet known."""
        fund = make_fund(n_quarters=4, lag_days=45)
        q_end = quarters(4)[-1]
        got = F.as_of(fund, q_end)
        assert q_end not in set(got["period_end"])

    def test_late_filer_is_not_used_early(self):
        """A filing broadcast two years late must be invisible until then."""
        fund = make_fund(n_quarters=4, lag_days=45)
        late = fund.iloc[[0]].copy()
        late["period_end"] = pd.Timestamp("2021-06-30")
        late["broadcast_date"] = pd.Timestamp("2023-05-15")
        fund = pd.concat([fund, late], ignore_index=True)

        mid = F.as_of(fund, pd.Timestamp("2022-01-01"))
        if not mid.empty:
            assert pd.Timestamp("2021-06-30") not in set(mid["period_end"])

    def test_only_most_recent_broadcast_survives(self):
        """A revision supersedes the original, from its own broadcast onward."""
        fund = make_fund(isins=("INE000A01001",), n_quarters=4)
        q = quarters(4)[-1]
        revision = fund[fund["period_end"] == q].copy()
        revision["broadcast_date"] = q + pd.Timedelta(days=90)
        revision["pat"] = 999.0
        fund = pd.concat([fund, revision], ignore_index=True)

        before = F.as_of(fund, q + pd.Timedelta(days=60))
        assert before.loc["INE000A01001", "pat"] == 100.0, "revision used before broadcast"

        after = F.as_of(fund, q + pd.Timedelta(days=100))
        assert after.loc["INE000A01001", "pat"] == 999.0, "revision should supersede"

    def test_stale_filings_are_dropped(self):
        """A company that stopped reporting must not contribute forever."""
        fund = make_fund(isins=("INE000A01001",), n_quarters=4)
        last = fund["period_end"].max()
        far_later = last + pd.Timedelta(days=1000)
        assert F.as_of(fund, far_later, max_age_days=400).empty

    def test_truncation_invariance(self):
        """Removing future filings must not change what was visible earlier."""
        fund = make_fund(n_quarters=12)
        when = pd.Timestamp("2021-09-30")

        full = F.as_of(fund, when)
        trunc = F.as_of(fund[fund["broadcast_date"] <= when], when)
        pd.testing.assert_frame_equal(full, trunc)


class TestTrailingFourQuarters:
    def test_sums_four_quarters(self):
        fund = make_fund(isins=("INE000A01001",), n_quarters=8, pat=100.0)
        when = quarters(8)[-1] + pd.Timedelta(days=45)
        ttm = F.trailing_four_quarters(fund, when, "pat")
        assert ttm["INE000A01001"] == pytest.approx(400.0)

    def test_respects_broadcast_visibility(self):
        fund = make_fund(isins=("INE000A01001",), n_quarters=8, pat=100.0)
        # Before the newest filing is public, the trailing year ends a quarter back.
        when = quarters(8)[-1] + pd.Timedelta(days=10)
        ttm = F.trailing_four_quarters(fund, when, "pat")
        assert ttm["INE000A01001"] == pytest.approx(400.0)
        visible = fund[fund["broadcast_date"] <= when]
        assert visible["period_end"].max() == quarters(8)[-2]

    def test_revision_not_double_counted(self):
        fund = make_fund(isins=("INE000A01001",), n_quarters=8, pat=100.0)
        q = quarters(8)[-1]
        rev = fund[fund["period_end"] == q].copy()
        rev["broadcast_date"] = q + pd.Timedelta(days=50)
        rev["pat"] = 200.0
        fund = pd.concat([fund, rev], ignore_index=True)

        ttm = F.trailing_four_quarters(fund, q + pd.Timedelta(days=60), "pat")
        # 3 old quarters at 100 + the revised 200, not 100+200 for the same quarter
        assert ttm["INE000A01001"] == pytest.approx(500.0)

    def test_insufficient_quarters_excluded(self):
        fund = make_fund(isins=("INE000A01001",), n_quarters=3)
        when = quarters(3)[-1] + pd.Timedelta(days=45)
        assert "INE000A01001" not in F.trailing_four_quarters(fund, when, "pat").index

    def test_gappy_history_excluded(self):
        """Four scattered quarters spanning years must not be summed as a year."""
        fund = make_fund(isins=("INE000A01001",), n_quarters=4)
        fund.loc[fund.index[0], "period_end"] = pd.Timestamp("2015-03-31")
        fund.loc[fund.index[0], "broadcast_date"] = pd.Timestamp("2015-05-15")
        when = pd.Timestamp("2021-06-30")
        assert "INE000A01001" not in F.trailing_four_quarters(fund, when, "pat").index


class TestXbrlParsing:
    def _doc(self, q_rev: float, ytd_rev: float) -> bytes:
        """Minimal XBRL where contextRef dates lie but reported dates do not.

        This mirrors the real filings: the quarterly and year-to-date contexts
        carry identical contextRef periods, and only the reported
        DateOfStartOfReportingPeriod distinguishes them.
        """
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:in="http://x">
  <xbrli:context id="OneD"><xbrli:period>
    <xbrli:startDate>2024-07-01</xbrli:startDate><xbrli:endDate>2024-09-30</xbrli:endDate>
  </xbrli:period></xbrli:context>
  <xbrli:context id="FourD"><xbrli:period>
    <xbrli:startDate>2024-07-01</xbrli:startDate><xbrli:endDate>2024-09-30</xbrli:endDate>
  </xbrli:period></xbrli:context>
  <xbrli:context id="OneI"><xbrli:period>
    <xbrli:instant>2024-09-30</xbrli:instant>
  </xbrli:period></xbrli:context>
  <in:DateOfStartOfReportingPeriod contextRef="OneD">2024-07-01</in:DateOfStartOfReportingPeriod>
  <in:DateOfEndOfReportingPeriod contextRef="OneD">2024-09-30</in:DateOfEndOfReportingPeriod>
  <in:DateOfStartOfReportingPeriod contextRef="FourD">2024-04-01</in:DateOfStartOfReportingPeriod>
  <in:DateOfEndOfReportingPeriod contextRef="FourD">2024-09-30</in:DateOfEndOfReportingPeriod>
  <in:RevenueFromOperations contextRef="OneD">{q_rev}</in:RevenueFromOperations>
  <in:RevenueFromOperations contextRef="FourD">{ytd_rev}</in:RevenueFromOperations>
  <in:ProfitLossForPeriod contextRef="OneD">1000</in:ProfitLossForPeriod>
  <in:PaidUpValueOfEquityShareCapital contextRef="OneD">81309000</in:PaidUpValueOfEquityShareCapital>
  <in:FaceValueOfEquityShareCapital contextRef="OneD">5</in:FaceValueOfEquityShareCapital>
  <in:EquityShareCapital contextRef="OneI">81309000</in:EquityShareCapital>
  <in:OtherEquity contextRef="OneI">5823219000</in:OtherEquity>
  <in:BorrowingsNoncurrent contextRef="OneI">1345994000</in:BorrowingsNoncurrent>
  <in:BorrowingsCurrent contextRef="OneI">1997451000</in:BorrowingsCurrent>
</xbrli:xbrl>""".encode()

    def test_picks_quarterly_not_ytd(self):
        """The trap: identical contextRef dates, different true periods."""
        got = F.parse_xbrl(self._doc(q_rev=4_637_484_000, ytd_rev=9_037_332_000))
        assert got is not None
        assert got["revenue"] == pytest.approx(4_637_484_000), "picked the YTD figure"
        assert got["period_start"] == pd.Timestamp("2024-07-01")
        assert got["quarter_days"] == 91

    def test_derives_shares_outstanding(self):
        got = F.parse_xbrl(self._doc(1e9, 2e9))
        assert got["shares_outstanding"] == pytest.approx(81_309_000 / 5)

    def test_derives_net_worth_and_debt(self):
        got = F.parse_xbrl(self._doc(1e9, 2e9))
        assert got["net_worth"] == pytest.approx(81_309_000 + 5_823_219_000)
        assert got["total_debt"] == pytest.approx(1_345_994_000 + 1_997_451_000)

    def test_malformed_document_returns_none(self):
        assert F.parse_xbrl(b"not xml at all") is None

    def test_missing_reporting_dates_returns_none(self):
        doc = b"""<?xml version="1.0"?><xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
        xmlns:in="http://x"><in:RevenueFromOperations contextRef="OneD">5</in:RevenueFromOperations>
        </xbrli:xbrl>"""
        assert F.parse_xbrl(doc) is None


class TestFundamentalFactors:
    def _panel(self, price: float = 10.0, when: str = "2021-06-30") -> pd.DataFrame:
        rows = []
        for isin in ("INE000A01001", "INE000A01002"):
            for d in pd.bdate_range("2021-06-01", when):
                rows.append({"date": d, "isin": isin, "symbol": "X", "close": price,
                             "adj_close": price, "turnover": 1e8})
        return pd.DataFrame(rows)

    def test_market_cap_uses_current_price(self):
        fund = make_fund(n_quarters=8, shares=1_000_000)
        panel = self._panel(price=10.0)
        when = pd.Timestamp("2021-06-30")
        mcap = FF.market_caps(panel, fund, when)
        assert mcap.iloc[0] == pytest.approx(10.0 * 1_000_000)

    def test_earnings_yield_is_ttm_over_mcap(self):
        fund = make_fund(isins=("INE000A01001",), n_quarters=8, pat=100.0, shares=1_000_000)
        panel = self._panel(price=10.0)
        when = pd.Timestamp("2021-06-30")
        got = FF.compute(panel, fund, when)
        expected = 400.0 / (10.0 * 1_000_000)
        assert got.loc["INE000A01001", "earnings_yield"] == pytest.approx(expected)

    def test_negative_net_worth_excluded_from_value(self):
        """Accumulated losses are not cheapness."""
        fund = make_fund(isins=("INE000A01001",), n_quarters=8, net_worth=-500.0)
        panel = self._panel()
        got = FF.compute(panel, fund, pd.Timestamp("2021-06-30"))
        assert np.isnan(got.loc["INE000A01001", "book_to_price"])
        assert np.isnan(got.loc["INE000A01001", "roe"])

    def test_factors_are_causal(self):
        """Future filings cannot change today's factor values."""
        fund = make_fund(n_quarters=12)
        panel = self._panel(when="2021-06-30")
        when = pd.Timestamp("2021-06-30")

        before = FF.compute(panel, fund, when)
        tampered = fund.copy()
        future = tampered["broadcast_date"] > when
        tampered.loc[future, ["pat", "net_worth"]] *= 50.0
        after = FF.compute(panel, tampered, when)

        pd.testing.assert_frame_equal(before, after)

    def test_earnings_stability_rewards_steadiness(self):
        steady = make_fund(isins=("INE000A01001",), n_quarters=10, pat=100.0)
        erratic = steady.copy()
        erratic["isin"] = "INE000A01002"
        rng = np.random.default_rng(0)
        erratic["pat"] = 100.0 + rng.normal(0, 80, len(erratic))
        fund = pd.concat([steady, erratic], ignore_index=True)

        when = fund["broadcast_date"].max()
        s = FF._earnings_stability(fund, when)
        assert s["INE000A01001"] > s["INE000A01002"]

    def test_all_declared_factors_are_produced(self):
        fund = make_fund(n_quarters=12)
        panel = self._panel()
        got = FF.compute(panel, fund, pd.Timestamp("2021-06-30"))
        missing = set(FF.FUND_FACTOR_SIGNS) - set(got.columns)
        assert not missing, f"declared but not computed: {missing}"

    def test_perfectly_stable_earnings_score_highest_not_nan(self):
        """Zero variance must be the best score, never a missing one."""
        fund = make_fund(isins=("INE000A01001",), n_quarters=10, pat=100.0)
        when = fund["broadcast_date"].max()
        s = FF._earnings_stability(fund, when)
        assert "INE000A01001" in s.index
        assert s["INE000A01001"] == pytest.approx(1.0 / FF.CV_FLOOR)
