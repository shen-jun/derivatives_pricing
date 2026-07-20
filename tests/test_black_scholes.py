"""
My unit tests for the Black-Scholes-Merton model. I check against a known
textbook value, verify put-call parity holds exactly (since it is an
algebraic identity, not an approximation), and verify my analytic Greeks
line up with a manual bump-and-reprice finite difference.
"""

import math

from src.models.black_scholes import BlackScholesModel


def test_known_call_price_matches_textbook_value():
    # Hull's textbook example: S=42, K=40, T=0.5, r=0.10, sigma=0.20, q=0
    # gives a call price close to 4.76.
    model = BlackScholesModel(spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10, volatility=0.20, option_type="call")
    assert math.isclose(model.price(), 4.76, abs_tol=0.02)


def test_known_put_price_matches_textbook_value():
    model = BlackScholesModel(spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10, volatility=0.20, option_type="put")
    assert math.isclose(model.price(), 0.81, abs_tol=0.02)


def test_put_call_parity_holds_exactly():
    spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield = 100, 105, 0.75, 0.03, 0.25, 0.01

    call_model = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, "call")
    put_model = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, "put")

    left_hand_side = call_model.price() - put_model.price()
    right_hand_side = spot * math.exp(-dividend_yield * time_to_expiry) - strike * math.exp(-risk_free_rate * time_to_expiry)

    assert math.isclose(left_hand_side, right_hand_side, abs_tol=1e-8)


def test_delta_matches_bump_and_reprice():
    spot, strike, time_to_expiry, risk_free_rate, volatility = 100, 100, 1.0, 0.05, 0.20
    model = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type="call")

    bump = 0.01
    model_up = BlackScholesModel(spot + bump, strike, time_to_expiry, risk_free_rate, volatility, option_type="call")
    model_down = BlackScholesModel(spot - bump, strike, time_to_expiry, risk_free_rate, volatility, option_type="call")
    bumped_delta = (model_up.price() - model_down.price()) / (2 * bump)

    assert math.isclose(model.delta(), bumped_delta, abs_tol=1e-4)


def test_gamma_matches_bump_and_reprice():
    spot, strike, time_to_expiry, risk_free_rate, volatility = 100, 100, 1.0, 0.05, 0.20
    model = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type="put")

    bump = 0.5
    model_up = BlackScholesModel(spot + bump, strike, time_to_expiry, risk_free_rate, volatility, option_type="put")
    model_down = BlackScholesModel(spot - bump, strike, time_to_expiry, risk_free_rate, volatility, option_type="put")
    bumped_gamma = (model_up.price() - 2 * model.price() + model_down.price()) / (bump ** 2)

    assert math.isclose(model.gamma(), bumped_gamma, abs_tol=1e-3)


def test_call_price_is_within_no_arbitrage_bounds():
    spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield = 100, 90, 1.0, 0.05, 0.30, 0.02
    model = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, "call")

    discounted_spot = spot * math.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * math.exp(-risk_free_rate * time_to_expiry)
    lower_bound = max(discounted_spot - discounted_strike, 0.0)
    upper_bound = discounted_spot

    price = model.price()
    assert lower_bound - 1e-8 <= price <= upper_bound + 1e-8


def test_deep_in_the_money_call_delta_approaches_one():
    model = BlackScholesModel(spot=300, strike=100, time_to_expiry=1.0, risk_free_rate=0.03, volatility=0.20, option_type="call")
    assert model.delta() > 0.99


def test_deep_out_of_the_money_put_delta_approaches_zero():
    model = BlackScholesModel(spot=300, strike=100, time_to_expiry=1.0, risk_free_rate=0.03, volatility=0.20, option_type="put")
    assert model.delta() > -0.01
