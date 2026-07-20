"""
This is my model validation suite, corresponding to Phase 12 of my project
specification. Unlike my other test files, which each focus on one model,
this file specifically targets the cross-cutting checks that any credible
model validation group would run before signing off on a pricing library:
put-call parity, no-arbitrage bounds, American >= European, tree
convergence, Monte Carlo convergence, Greek stability, and correct barrier
behavior.
"""

import math

from src.models.black_scholes import BlackScholesModel
from src.models.binomial_tree import CRRBinomialTree, LeisenReimerTree
from src.models.monte_carlo import AsianOption
from src.models.finite_difference import FiniteDifferencePDE


def test_put_call_parity_black_scholes():
    spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield = 120, 100, 1.5, 0.04, 0.22, 0.015

    call_price = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, "call").price()
    put_price = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, "put").price()

    left_hand_side = call_price - put_price
    right_hand_side = spot * math.exp(-dividend_yield * time_to_expiry) - strike * math.exp(-risk_free_rate * time_to_expiry)
    assert math.isclose(left_hand_side, right_hand_side, abs_tol=1e-8)


def test_no_arbitrage_bounds_black_scholes_call_and_put():
    spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield = 100, 110, 1.0, 0.05, 0.30, 0.0

    call_price = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, "call").price()
    put_price = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, "put").price()

    discounted_spot = spot * math.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * math.exp(-risk_free_rate * time_to_expiry)

    assert max(discounted_spot - discounted_strike, 0.0) - 1e-8 <= call_price <= discounted_spot + 1e-8
    assert max(discounted_strike - discounted_spot, 0.0) - 1e-8 <= put_price <= discounted_strike + 1e-8


def test_american_price_is_never_below_european_price_crr():
    params = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.06, volatility=0.25, dividend_yield=0.03)

    for option_type in ("call", "put"):
        american_price = CRRBinomialTree(option_type=option_type, steps=250, american=True, **params).price()
        european_price = CRRBinomialTree(option_type=option_type, steps=250, american=False, **params).price()
        assert american_price >= european_price - 1e-8


def test_crr_convergence_error_shrinks_as_steps_increase():
    params = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.0, option_type="call")
    benchmark_price = BlackScholesModel(**params).price()

    error_at_few_steps = abs(CRRBinomialTree(steps=25, american=False, **params).price() - benchmark_price)
    error_at_many_steps = abs(CRRBinomialTree(steps=400, american=False, **params).price() - benchmark_price)

    assert error_at_many_steps < error_at_few_steps


def test_leisen_reimer_convergence_error_shrinks_as_steps_increase():
    params = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.0, option_type="put")
    benchmark_price = BlackScholesModel(**params).price()

    error_at_few_steps = abs(LeisenReimerTree(steps=25, american=False, **params).price() - benchmark_price)
    error_at_many_steps = abs(LeisenReimerTree(steps=201, american=False, **params).price() - benchmark_price)

    assert error_at_many_steps < error_at_few_steps


def test_monte_carlo_standard_error_shrinks_with_more_paths():
    params = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.03, volatility=0.25, dividend_yield=0.0, option_type="call")

    model_few_paths = AsianOption(num_paths=500, num_steps=20, random_seed=42, **params)
    model_many_paths = AsianOption(num_paths=8000, num_steps=20, random_seed=42, **params)

    model_few_paths.price()
    model_many_paths.price()

    assert model_many_paths._last_std_error < model_few_paths._last_std_error


def test_greeks_are_stable_under_a_small_input_perturbation():
    """
    A well-behaved pricing model should not produce wildly different Greeks
    for a tiny change in the underlying price. I check that delta does not
    jump by an unreasonable amount when I nudge spot by a small fraction of
    a percent.
    """
    base_model = BlackScholesModel(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.20, option_type="call")
    nudged_model = BlackScholesModel(spot=100.05, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.20, option_type="call")

    delta_change = abs(nudged_model.delta() - base_model.delta())
    assert delta_change < 0.01


def test_down_and_out_barrier_price_is_bounded_by_vanilla_price():
    """
    In theory a knock-out barrier option can never be worth more than the
    corresponding vanilla option. In practice, my explicit finite difference
    scheme carries its own discretization error (I only use a first-order
    accurate scheme, on purpose, to keep the update formula readable), so I
    allow a small numerical tolerance here rather than an exact bound. I use
    a moderately fine grid (150 price steps) to keep that numerical error
    small.
    """
    params = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.03, volatility=0.20, dividend_yield=0.0, option_type="call")

    vanilla_price = BlackScholesModel(**params).price()
    barrier_price = FiniteDifferencePDE(barrier_level=75, barrier_type="down-and-out", num_price_steps=150, num_time_steps=450, **params).price()

    assert 0.0 <= barrier_price <= vanilla_price + 0.05


def test_barrier_at_the_money_knocks_out_almost_all_value():
    """
    If I set a down-and-out barrier essentially at the current spot price,
    the option should be worth close to nothing, since it is immediately at
    risk of being knocked out.
    """
    params = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.03, volatility=0.20, dividend_yield=0.0, option_type="call")

    barrier_price = FiniteDifferencePDE(barrier_level=99.5, barrier_type="down-and-out", num_price_steps=80, num_time_steps=200, **params).price()
    vanilla_price = BlackScholesModel(**params).price()

    assert barrier_price < 0.5 * vanilla_price
