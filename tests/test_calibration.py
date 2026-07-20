"""
My unit tests for the implied volatility solver. The cleanest way to test a
root finder is to price an option at a volatility I already know, then
check that the solver recovers that same volatility from the resulting
price.
"""

import math

from src.models.black_scholes import BlackScholesModel
from src.calibration.implied_volatility import ImpliedVolatilitySolver


def test_solver_recovers_known_volatility_for_call():
    true_volatility = 0.27
    spot, strike, time_to_expiry, risk_free_rate, dividend_yield = 100, 105, 0.75, 0.04, 0.01

    market_price = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, true_volatility, dividend_yield, "call").price()

    solver = ImpliedVolatilitySolver()
    recovered_volatility = solver.solve(
        market_price, spot, strike, time_to_expiry, risk_free_rate, dividend_yield, "call", initial_guess=0.20
    )

    assert math.isclose(recovered_volatility, true_volatility, abs_tol=1e-4)


def test_solver_recovers_known_volatility_for_put():
    true_volatility = 0.18
    spot, strike, time_to_expiry, risk_free_rate, dividend_yield = 100, 90, 0.50, 0.03, 0.0

    market_price = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, true_volatility, dividend_yield, "put").price()

    solver = ImpliedVolatilitySolver()
    recovered_volatility = solver.solve(
        market_price, spot, strike, time_to_expiry, risk_free_rate, dividend_yield, "put", initial_guess=0.35
    )

    assert math.isclose(recovered_volatility, true_volatility, abs_tol=1e-4)


def test_solver_rejects_price_outside_no_arbitrage_bounds():
    import pytest

    solver = ImpliedVolatilitySolver()
    with pytest.raises(ValueError):
        solver.solve(
            market_price=1000.0, spot=100, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, dividend_yield=0.0, option_type="call",
        )


def test_solver_falls_back_to_bisection_from_a_poor_initial_guess():
    true_volatility = 0.40
    spot, strike, time_to_expiry, risk_free_rate, dividend_yield = 100, 100, 1.0, 0.02, 0.0

    market_price = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, true_volatility, dividend_yield, "call").price()

    solver = ImpliedVolatilitySolver()
    recovered_volatility = solver.solve(
        market_price, spot, strike, time_to_expiry, risk_free_rate, dividend_yield, "call", initial_guess=4.99,
    )

    assert math.isclose(recovered_volatility, true_volatility, abs_tol=1e-3)
