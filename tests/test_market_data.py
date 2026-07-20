"""
My unit tests for the market data container. I only test the manual
builder path here (not the live Yahoo Finance fetcher), since my test suite
needs to run in environments without outbound network access.
"""

from src.market_data.data_fetcher import build_manual_market_data_point


def test_manual_market_data_point_stores_all_fields():
    data_point = build_manual_market_data_point(
        underlying_price=100.0,
        strike_price=105.0,
        expiry_date="2026-12-18",
        risk_free_rate=0.05,
        dividend_yield=0.01,
        implied_volatility=0.22,
        market_option_price=4.50,
        bid=4.40,
        ask=4.60,
        volume=1200,
        open_interest=5400,
    )

    assert data_point.underlying_price == 100.0
    assert data_point.strike_price == 105.0
    assert data_point.expiry_date == "2026-12-18"
    assert data_point.implied_volatility == 0.22


def test_bid_ask_spread_is_computed_correctly():
    data_point = build_manual_market_data_point(
        underlying_price=100.0, strike_price=100.0, expiry_date="2026-06-19",
        risk_free_rate=0.04, bid=2.10, ask=2.30,
    )
    assert abs(data_point.bid_ask_spread - 0.20) < 1e-8


def test_bid_ask_spread_is_none_when_missing():
    data_point = build_manual_market_data_point(
        underlying_price=100.0, strike_price=100.0, expiry_date="2026-06-19", risk_free_rate=0.04,
    )
    assert data_point.bid_ask_spread is None
