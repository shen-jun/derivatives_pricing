"""
My unit tests for the scenario analysis engine. I check that the shock
reports contain the exact shock sizes specified in Phase 9 of my project
specification, and that the sign of the resulting P&L makes economic sense
for a simple call option.
"""

from src.models.black_scholes import BlackScholesModel
from src.scenario_analysis.scenario_engine import ScenarioEngine


def build_test_model():
    return BlackScholesModel(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.20, dividend_yield=0.01, option_type="call")


def test_underlying_price_shock_report_has_expected_shock_sizes():
    engine = ScenarioEngine(build_test_model())
    rows = engine.underlying_price_shock_report()

    shock_sizes = sorted(row["shock_pct"] for row in rows)
    assert shock_sizes == sorted([1.0, -1.0, 5.0, -5.0, 10.0, -10.0])


def test_call_price_increases_when_spot_shocked_upward():
    engine = ScenarioEngine(build_test_model())
    rows = engine.underlying_price_shock_report()

    for row in rows:
        if row["shock_pct"] > 0:
            assert row["pnl"] > 0
        else:
            assert row["pnl"] < 0


def test_volatility_shock_report_has_expected_shock_sizes():
    engine = ScenarioEngine(build_test_model())
    rows = engine.volatility_shock_report()

    shock_sizes = sorted(row["vol_shock_points"] for row in rows)
    assert shock_sizes == sorted([5.0, -5.0, 10.0, -10.0])


def test_call_price_increases_when_volatility_shocked_upward():
    engine = ScenarioEngine(build_test_model())
    rows = engine.volatility_shock_report()

    for row in rows:
        if row["vol_shock_points"] > 0:
            assert row["pnl"] > 0
        else:
            assert row["pnl"] < 0


def test_rate_shock_report_has_expected_shock_sizes():
    engine = ScenarioEngine(build_test_model())
    rows = engine.rate_shock_report()

    shock_sizes = sorted(row["rate_shock_bps"] for row in rows)
    assert shock_sizes == sorted([50, -50, 100, -100])


def test_time_decay_report_has_expected_day_offsets():
    engine = ScenarioEngine(build_test_model())
    rows = engine.time_decay_report()

    day_offsets = sorted(row["days_forward"] for row in rows)
    assert day_offsets == [1, 7, 30]


def test_full_report_excludes_correlation_and_barrier_for_vanilla_model():
    engine = ScenarioEngine(build_test_model())
    report = engine.full_report()

    assert "correlation_shocks" not in report
    assert "barrier_shocks" not in report
    assert set(report.keys()) == {"underlying_price_shocks", "volatility_shocks", "rate_shocks", "time_decay"}
