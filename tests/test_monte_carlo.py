"""
My unit tests for the Monte Carlo engine: Asian options, Lookback options,
and Basket options. I keep num_paths and num_steps modest in these tests
since my Monte Carlo engine uses explicit Python loops rather than
vectorized numpy, and I want the test suite to run quickly.
"""

import math

from src.models.black_scholes import BlackScholesModel
from src.models.monte_carlo import AsianOption, LookbackOption, BasketOption


def test_asian_call_price_is_cheaper_than_vanilla_call():
    """
    Averaging the path reduces the effective volatility of the payoff, so
    an arithmetic Asian call should always be worth less than a European
    vanilla call with the same strike.
    """
    params = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.03, volatility=0.30, dividend_yield=0.0)

    vanilla_price = BlackScholesModel(option_type="call", **params).price()
    asian_model = AsianOption(option_type="call", num_paths=4000, num_steps=30, random_seed=7, **params)
    asian_price = asian_model.price()

    assert asian_price < vanilla_price
    assert asian_price > 0


def test_asian_confidence_interval_brackets_the_price():
    params = dict(spot=100, strike=95, time_to_expiry=0.5, risk_free_rate=0.02, volatility=0.25, dividend_yield=0.0)
    model = AsianOption(option_type="call", num_paths=3000, num_steps=20, random_seed=1, **params)

    price = model.price()
    lower_bound, upper_bound = model.confidence_interval()

    assert lower_bound <= price <= upper_bound


def test_more_paths_reduces_the_confidence_interval_width():
    params = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.03, volatility=0.30, dividend_yield=0.0)

    small_sample_model = AsianOption(option_type="call", num_paths=500, num_steps=20, random_seed=3, **params)
    large_sample_model = AsianOption(option_type="call", num_paths=8000, num_steps=20, random_seed=3, **params)

    small_lower, small_upper = small_sample_model.confidence_interval()
    large_lower, large_upper = large_sample_model.confidence_interval()

    assert (large_upper - large_lower) < (small_upper - small_lower)


def test_fixed_strike_lookback_call_is_worth_at_least_vanilla_call():
    """
    max(path) >= S(T) always, so max(max(path) - K, 0) >= max(S(T) - K, 0)
    on every single path, which means the lookback call price must be at
    least as large as the vanilla call price.
    """
    params = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.03, volatility=0.25, dividend_yield=0.0)

    vanilla_price = BlackScholesModel(option_type="call", **params).price()
    lookback_model = LookbackOption(option_type="call", num_paths=4000, num_steps=30, random_seed=11, floating_strike=False, **params)
    lookback_price = lookback_model.price()

    assert lookback_price >= vanilla_price - 0.20  # small tolerance for MC noise


def test_basket_option_price_is_positive_and_finite():
    model = BasketOption(
        spots=[100, 95],
        weights=[0.5, 0.5],
        volatilities=[0.2, 0.25],
        correlation_matrix=[[1.0, 0.4], [0.4, 1.0]],
        strike=100,
        time_to_expiry=1.0,
        risk_free_rate=0.03,
        dividend_yields=[0.0, 0.0],
        option_type="call",
        num_paths=3000,
        num_steps=20,
        random_seed=5,
    )

    price = model.price()
    assert price > 0
    assert math.isfinite(price)


def test_basket_call_price_increases_with_correlation():
    """
    A basket call option should become more valuable as the correlation
    between its constituents rises, because a more correlated basket has a
    wider dispersion of terminal outcomes (more upside scenarios where the
    whole basket rallies together).
    """
    common_params = dict(
        spots=[100, 100],
        weights=[0.5, 0.5],
        volatilities=[0.25, 0.25],
        strike=100,
        time_to_expiry=1.0,
        risk_free_rate=0.03,
        dividend_yields=[0.0, 0.0],
        option_type="call",
        num_paths=3000,
        num_steps=20,
        random_seed=9,
    )

    low_correlation_model = BasketOption(correlation_matrix=[[1.0, 0.1], [0.1, 1.0]], **common_params)
    high_correlation_model = BasketOption(correlation_matrix=[[1.0, 0.9], [0.9, 1.0]], **common_params)

    assert high_correlation_model.price() >= low_correlation_model.price() - 0.10


def test_correlation_sensitivity_is_positive_for_basket_call():
    model = BasketOption(
        spots=[100, 100],
        weights=[0.5, 0.5],
        volatilities=[0.25, 0.25],
        correlation_matrix=[[1.0, 0.3], [0.3, 1.0]],
        strike=100,
        time_to_expiry=1.0,
        risk_free_rate=0.03,
        dividend_yields=[0.0, 0.0],
        option_type="call",
        num_paths=3000,
        num_steps=20,
        random_seed=13,
    )

    pnl_from_correlation_bump = model.correlation_sensitivity(correlation_bump=0.30)
    assert pnl_from_correlation_bump >= -0.10
