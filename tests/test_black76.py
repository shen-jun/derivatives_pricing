"""
My unit tests for the Black-76 model, used for options on futures.
"""

import math

from src.models.black76 import Black76Model


def test_put_call_parity_holds_for_futures_options():
    futures_price, strike, time_to_expiry, risk_free_rate, volatility = 50, 52, 0.5, 0.04, 0.25

    call_model = Black76Model(futures_price, strike, time_to_expiry, risk_free_rate, volatility, option_type="call")
    put_model = Black76Model(futures_price, strike, time_to_expiry, risk_free_rate, volatility, option_type="put")

    left_hand_side = call_model.price() - put_model.price()
    right_hand_side = math.exp(-risk_free_rate * time_to_expiry) * (futures_price - strike)

    assert math.isclose(left_hand_side, right_hand_side, abs_tol=1e-8)


def test_at_the_money_call_and_put_have_equal_delta_magnitude_relationship():
    futures_price, strike, time_to_expiry, risk_free_rate, volatility = 100, 100, 1.0, 0.03, 0.20

    call_model = Black76Model(futures_price, strike, time_to_expiry, risk_free_rate, volatility, option_type="call")
    put_model = Black76Model(futures_price, strike, time_to_expiry, risk_free_rate, volatility, option_type="put")

    discount_factor = math.exp(-risk_free_rate * time_to_expiry)
    assert math.isclose(call_model.delta() - put_model.delta(), discount_factor, abs_tol=1e-8)


def test_rho_equals_negative_time_times_price():
    futures_price, strike, time_to_expiry, risk_free_rate, volatility = 100, 100, 2.0, 0.05, 0.20
    model = Black76Model(futures_price, strike, time_to_expiry, risk_free_rate, volatility, option_type="call")

    assert math.isclose(model.rho(), -time_to_expiry * model.price(), abs_tol=1e-8)


def test_vega_matches_bump_and_reprice():
    futures_price, strike, time_to_expiry, risk_free_rate, volatility = 100, 105, 0.75, 0.02, 0.30
    model = Black76Model(futures_price, strike, time_to_expiry, risk_free_rate, volatility, option_type="put")

    bump = 0.0005
    model_up = Black76Model(futures_price, strike, time_to_expiry, risk_free_rate, volatility + bump, option_type="put")
    model_down = Black76Model(futures_price, strike, time_to_expiry, risk_free_rate, volatility - bump, option_type="put")
    bumped_vega = (model_up.price() - model_down.price()) / (2 * bump)

    assert math.isclose(model.vega(), bumped_vega, abs_tol=1e-3)
