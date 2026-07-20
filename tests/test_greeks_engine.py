"""
My unit tests for the Greeks aggregation engine.
"""

from src.models.black_scholes import BlackScholesModel
from src.greeks.greeks_engine import GreeksEngine


def test_greeks_table_has_expected_columns_and_rows():
    engine = GreeksEngine()

    call_model = BlackScholesModel(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.20, option_type="call")
    put_model = BlackScholesModel(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.20, option_type="put")

    table = engine.build_greeks_table({"call": call_model, "put": put_model})

    assert list(table.columns) == ["price", "delta", "gamma", "vega", "theta", "rho"]
    assert set(table.index) == {"call", "put"}
    assert table.loc["call", "delta"] > table.loc["put", "delta"]


def test_heatmap_grid_has_correct_shape():
    engine = GreeksEngine()

    def model_factory(spot, volatility):
        return BlackScholesModel(spot=spot, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=volatility, option_type="call")

    spot_values = [90, 100, 110]
    vol_values = [0.15, 0.25]

    grid = engine.build_heatmap_grid(model_factory, spot_values, vol_values, greek_name="delta")

    assert len(grid) == len(vol_values)
    for row in grid:
        assert len(row) == len(spot_values)


def test_sensitivity_report_base_price_matches_model_price():
    engine = GreeksEngine()
    model = BlackScholesModel(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.20, option_type="call")

    report = engine.build_sensitivity_report(model)
    base_price_row = report[report["risk_factor"] == "base_price"]

    assert abs(base_price_row["greek_implied_pnl"].iloc[0] - model.price()) < 1e-8
