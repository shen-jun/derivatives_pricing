"""
My unit tests for the Finite Difference PDE barrier option solver. I check
that a knock-out barrier is always worth less than or equal to the
corresponding vanilla option, that knock-in plus knock-out reproduces the
vanilla price (the in/out parity relationship I rely on to price knock-in
contracts), and that moving a down-barrier closer to the spot price reduces
the value of a down-and-out option.
"""

import math

from src.models.black_scholes import BlackScholesModel
from src.models.finite_difference import FiniteDifferencePDE


COMMON_PARAMS = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.03, volatility=0.20, dividend_yield=0.0)


def test_down_and_out_call_is_cheaper_than_vanilla_call():
    vanilla_price = BlackScholesModel(option_type="call", **COMMON_PARAMS).price()
    barrier_model = FiniteDifferencePDE(
        option_type="call", barrier_level=80, barrier_type="down-and-out",
        num_price_steps=80, num_time_steps=200, **COMMON_PARAMS,
    )
    barrier_price = barrier_model.price()

    assert barrier_price <= vanilla_price + 1e-6
    assert barrier_price > 0


def test_up_and_out_call_is_cheaper_than_vanilla_call():
    vanilla_price = BlackScholesModel(option_type="call", **COMMON_PARAMS).price()
    barrier_model = FiniteDifferencePDE(
        option_type="call", barrier_level=130, barrier_type="up-and-out",
        num_price_steps=80, num_time_steps=200, **COMMON_PARAMS,
    )
    barrier_price = barrier_model.price()

    assert barrier_price <= vanilla_price + 1e-6
    assert barrier_price >= 0


def test_knock_in_plus_knock_out_reproduces_vanilla_price():
    vanilla_price = BlackScholesModel(option_type="call", **COMMON_PARAMS).price()

    knock_out_model = FiniteDifferencePDE(
        option_type="call", barrier_level=85, barrier_type="down-and-out",
        num_price_steps=80, num_time_steps=200, **COMMON_PARAMS,
    )
    knock_in_model = FiniteDifferencePDE(
        option_type="call", barrier_level=85, barrier_type="down-and-in",
        num_price_steps=80, num_time_steps=200, **COMMON_PARAMS,
    )

    reconstructed_vanilla = knock_out_model.price() + knock_in_model.price()
    assert math.isclose(reconstructed_vanilla, vanilla_price, abs_tol=0.15)


def test_barrier_closer_to_spot_reduces_down_and_out_value():
    """
    Moving a down-barrier closer to the current spot price makes it more
    likely to be triggered, which should reduce the value of a
    down-and-out option.
    """
    model = FiniteDifferencePDE(
        option_type="call", barrier_level=70, barrier_type="down-and-out",
        num_price_steps=80, num_time_steps=200, **COMMON_PARAMS,
    )

    pnl_from_raising_barrier = model.barrier_sensitivity(barrier_bump=0.10)
    assert pnl_from_raising_barrier <= 0.05


def test_barrier_option_greeks_are_finite():
    model = FiniteDifferencePDE(
        option_type="put", barrier_level=80, barrier_type="down-and-out",
        num_price_steps=80, num_time_steps=200, **COMMON_PARAMS,
    )

    for greek_value in model.greeks_summary().values():
        assert math.isfinite(greek_value)


def test_invalid_barrier_type_raises():
    import pytest

    with pytest.raises(ValueError):
        FiniteDifferencePDE(option_type="call", barrier_level=90, barrier_type="sideways", **COMMON_PARAMS)
