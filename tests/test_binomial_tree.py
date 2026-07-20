"""
My unit tests for the CRR and Leisen-Reimer binomial trees. I check:
convergence to the Black-Scholes-Merton price when I disable early exercise
(american=False), that American prices are never below their European
counterparts, and that the LR tree traces out a sensible early exercise
boundary for an American put.
"""

import math

from src.models.black_scholes import BlackScholesModel
from src.models.binomial_tree import CRRBinomialTree, LeisenReimerTree


COMMON_PARAMS = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.02)


def test_crr_european_converges_to_black_scholes():
    bsm_price = BlackScholesModel(option_type="call", **COMMON_PARAMS).price()
    tree_price = CRRBinomialTree(option_type="call", steps=400, american=False, **COMMON_PARAMS).price()

    assert math.isclose(bsm_price, tree_price, abs_tol=0.05)


def test_leisen_reimer_european_converges_to_black_scholes():
    bsm_price = BlackScholesModel(option_type="put", **COMMON_PARAMS).price()
    tree_price = LeisenReimerTree(option_type="put", steps=201, american=False, **COMMON_PARAMS).price()

    assert math.isclose(bsm_price, tree_price, abs_tol=0.02)


def test_leisen_reimer_converges_faster_than_crr():
    """
    I check that at a moderate step count, the LR tree is at least as close
    (usually closer) to the true Black-Scholes price than CRR is, which is
    the entire point of using the LR probability transformation.
    """
    bsm_price = BlackScholesModel(option_type="call", **COMMON_PARAMS).price()

    crr_error = abs(CRRBinomialTree(option_type="call", steps=51, american=False, **COMMON_PARAMS).price() - bsm_price)
    lr_error = abs(LeisenReimerTree(option_type="call", steps=51, american=False, **COMMON_PARAMS).price() - bsm_price)

    assert lr_error <= crr_error + 1e-6


def test_american_put_price_is_at_least_european_put_price():
    american_put = CRRBinomialTree(option_type="put", steps=300, american=True, **COMMON_PARAMS).price()
    european_put = CRRBinomialTree(option_type="put", steps=300, american=False, **COMMON_PARAMS).price()

    assert american_put >= european_put - 1e-8


def test_american_call_with_dividends_can_exceed_european_call():
    """
    With a positive dividend yield, early exercise of an American call can
    become optimal, so I expect the American price to sit above (or equal
    to) the European price.
    """
    american_call = CRRBinomialTree(option_type="call", steps=300, american=True, **COMMON_PARAMS).price()
    european_call = CRRBinomialTree(option_type="call", steps=300, american=False, **COMMON_PARAMS).price()

    assert american_call >= european_call - 1e-8


def test_early_exercise_boundary_is_below_strike_for_american_put():
    tree = CRRBinomialTree(option_type="put", steps=100, american=True, **COMMON_PARAMS)
    boundary_points = tree.early_exercise_boundary()

    critical_prices = [price for _, price in boundary_points if price is not None]
    assert len(critical_prices) > 0
    for critical_price in critical_prices:
        assert critical_price <= COMMON_PARAMS["strike"]


def test_tree_raises_on_arbitrage_inducing_parameters():
    """
    If volatility is far too small relative to the drift over one time
    step, the risk-neutral probability can fall outside (0, 1). I check
    that my tree detects this rather than silently returning a bad number.
    """
    import pytest

    with pytest.raises(ValueError):
        CRRBinomialTree(
            spot=100, strike=100, time_to_expiry=5.0, risk_free_rate=0.9,
            volatility=0.001, dividend_yield=0.0, option_type="call", steps=5, american=False,
        ).price()
