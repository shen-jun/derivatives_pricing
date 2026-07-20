"""
This is my market data collection module, corresponding to Phase 1 of my
project specification. In a full production setup I would pull live data
from Yahoo Finance, CBOE, NASDAQ, Polygon, or OptionMetrics, but since this
project runs in an environment that may not always have outbound network
access, I built this module around a small, well-defined data container
(MarketDataPoint) plus a Yahoo Finance fetcher that degrades gracefully
when the network call fails or the `yfinance` package is not installed.

Every other part of my platform (pricing models, Greeks engine, scenario
engine, dashboard) only ever depends on MarketDataPoint, never on yfinance
directly. That way, if I ever want to plug in Polygon or CBOE instead, I
only have to write a new fetcher function that returns a MarketDataPoint,
and nothing downstream has to change.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketDataPoint:
    """
    My standard container for every market input I need to price and risk
    an option contract, matching the "Collect" list from Phase 1 of my
    project specification.
    """

    underlying_price: float
    strike_price: float
    expiry_date: str
    risk_free_rate: float
    dividend_yield: float
    implied_volatility: Optional[float] = None
    market_option_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    @property
    def bid_ask_spread(self):
        """I compute the bid-ask spread on demand rather than storing it separately."""
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


class YahooFinanceFetcher:
    """
    My thin wrapper around the `yfinance` package for pulling the
    underlying spot price and the listed option chain for a given ticker.

    I isolate the `import yfinance` call inside the constructor (rather than
    at module import time) so that the rest of my codebase can be imported
    and unit-tested even in an environment where `yfinance` is not
    installed or where outbound network calls are blocked.
    """

    def __init__(self):
        try:
            import yfinance  # noqa: F401
            self._yfinance = yfinance
        except ImportError as exc:
            raise ImportError(
                "I need the 'yfinance' package installed to fetch live market data. "
                "Install it with: pip install yfinance"
            ) from exc

    def fetch_spot_price(self, ticker_symbol):
        """I fetch the most recent close price for a given ticker symbol."""
        ticker = self._yfinance.Ticker(ticker_symbol)
        history = ticker.history(period="1d")
        if history.empty:
            raise RuntimeError(f"I could not retrieve any price history for ticker '{ticker_symbol}'.")
        return float(history["Close"].iloc[-1])

    def fetch_option_chain(self, ticker_symbol, expiry_date=None):
        """
        I fetch the full option chain (calls and puts) for a given ticker
        and expiry date, and translate every row into a MarketDataPoint so
        the rest of my platform never has to know the shape of the raw
        yfinance DataFrame.
        """
        ticker = self._yfinance.Ticker(ticker_symbol)
        available_expiries = ticker.options
        if not available_expiries:
            raise RuntimeError(f"I found no listed option expiries for ticker '{ticker_symbol}'.")

        if expiry_date is None:
            expiry_date = available_expiries[0]
        elif expiry_date not in available_expiries:
            raise ValueError(f"I could not find expiry_date '{expiry_date}' in the listed expiries.")

        option_chain = ticker.option_chain(expiry_date)
        spot_price = self.fetch_spot_price(ticker_symbol)

        market_data_points = []
        for _, row in option_chain.calls.iterrows():
            market_data_points.append(
                MarketDataPoint(
                    underlying_price=spot_price,
                    strike_price=float(row["strike"]),
                    expiry_date=expiry_date,
                    risk_free_rate=0.0,
                    dividend_yield=0.0,
                    implied_volatility=float(row.get("impliedVolatility", float("nan"))),
                    market_option_price=float(row.get("lastPrice", float("nan"))),
                    bid=float(row.get("bid", float("nan"))),
                    ask=float(row.get("ask", float("nan"))),
                    volume=int(row.get("volume", 0) or 0),
                    open_interest=int(row.get("openInterest", 0) or 0),
                )
            )

        return market_data_points


def build_manual_market_data_point(
    underlying_price,
    strike_price,
    expiry_date,
    risk_free_rate,
    dividend_yield=0.0,
    implied_volatility=None,
    market_option_price=None,
    bid=None,
    ask=None,
    volume=None,
    open_interest=None,
):
    """
    I use this helper throughout my tests, examples, and dashboard whenever
    I want to build a MarketDataPoint from numbers I typed in myself,
    instead of going through the Yahoo Finance fetcher. This is the code
    path I exercise in an environment with no network access.
    """
    return MarketDataPoint(
        underlying_price=underlying_price,
        strike_price=strike_price,
        expiry_date=expiry_date,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        implied_volatility=implied_volatility,
        market_option_price=market_option_price,
        bid=bid,
        ask=ask,
        volume=volume,
        open_interest=open_interest,
    )
