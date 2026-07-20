"""
This is my implied volatility solver. Given a market-observed option price, I
back out the volatility input that makes my Black-Scholes-Merton model
reproduce that market price exactly.

I use Newton-Raphson as my primary method because it converges very quickly
(vega is cheap to compute and the function is well-behaved for reasonable
inputs), but I fall back to a bisection search whenever Newton-Raphson fails
to converge or steps outside a sane volatility range. I added the bisection
fallback because Newton-Raphson can behave badly if I start too far from the
root or if vega happens to be extremely small (deep in-the-money or deep
out-of-the-money options).
"""

from src.models.black_scholes import BlackScholesModel


class ImpliedVolatilitySolver:
    """
    My implied volatility calibration engine for European options priced
    under Black-Scholes-Merton.
    """

    def __init__(self, max_iterations=100, tolerance=1e-8, min_volatility=1e-4, max_volatility=5.0):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility

    def _price_at_volatility(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type):
        model = BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type)
        return model.price(), model.vega()

    def solve(
        self,
        market_price,
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        dividend_yield=0.0,
        option_type="call",
        initial_guess=0.20,
    ):
        """
        I return the implied volatility that reprices `market_price` under
        my Black-Scholes-Merton model, using Newton-Raphson with a
        bisection fallback.
        """
        no_arbitrage_lower_bound, no_arbitrage_upper_bound = self._no_arbitrage_bounds(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, option_type
        )
        if market_price < no_arbitrage_lower_bound - 1e-8 or market_price > no_arbitrage_upper_bound + 1e-8:
            raise ValueError(
                "I cannot find an implied volatility for this market_price because it violates my "
                "no-arbitrage bounds for a European option with these inputs."
            )

        newton_result = self._newton_raphson(
            market_price, spot, strike, time_to_expiry, risk_free_rate, dividend_yield, option_type, initial_guess
        )
        if newton_result is not None:
            return newton_result

        return self._bisection(
            market_price, spot, strike, time_to_expiry, risk_free_rate, dividend_yield, option_type
        )

    def _newton_raphson(self, market_price, spot, strike, time_to_expiry, risk_free_rate, dividend_yield, option_type, initial_guess):
        volatility = initial_guess

        for _ in range(self.max_iterations):
            if volatility <= self.min_volatility or volatility >= self.max_volatility:
                return None

            model_price, model_vega = self._price_at_volatility(
                spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type
            )
            price_difference = model_price - market_price

            if abs(price_difference) < self.tolerance:
                return volatility

            if model_vega < 1e-8:
                return None

            volatility = volatility - price_difference / model_vega

        return None

    def _bisection(self, market_price, spot, strike, time_to_expiry, risk_free_rate, dividend_yield, option_type):
        low_volatility = self.min_volatility
        high_volatility = self.max_volatility

        low_price, _ = self._price_at_volatility(spot, strike, time_to_expiry, risk_free_rate, low_volatility, dividend_yield, option_type)
        high_price, _ = self._price_at_volatility(spot, strike, time_to_expiry, risk_free_rate, high_volatility, dividend_yield, option_type)

        if not (low_price <= market_price <= high_price):
            raise ValueError(
                "I could not bracket the market price between my min_volatility and max_volatility bounds."
            )

        for _ in range(200):
            mid_volatility = 0.5 * (low_volatility + high_volatility)
            mid_price, _ = self._price_at_volatility(
                spot, strike, time_to_expiry, risk_free_rate, mid_volatility, dividend_yield, option_type
            )

            if abs(mid_price - market_price) < self.tolerance:
                return mid_volatility

            if mid_price < market_price:
                low_volatility = mid_volatility
            else:
                high_volatility = mid_volatility

        return 0.5 * (low_volatility + high_volatility)

    @staticmethod
    def _no_arbitrage_bounds(spot, strike, time_to_expiry, risk_free_rate, dividend_yield, option_type):
        """
        I compute the standard no-arbitrage price bounds for a European
        option so I can reject a market price up front instead of letting
        my root finder search forever for a volatility that cannot exist.

        Call: max(S*exp(-qT) - K*exp(-rT), 0) <= Call <= S*exp(-qT)
        Put:  max(K*exp(-rT) - S*exp(-qT), 0) <= Put  <= K*exp(-rT)
        """
        import math

        discounted_spot = spot * math.exp(-dividend_yield * time_to_expiry)
        discounted_strike = strike * math.exp(-risk_free_rate * time_to_expiry)

        if option_type == "call":
            lower_bound = max(discounted_spot - discounted_strike, 0.0)
            upper_bound = discounted_spot
        else:
            lower_bound = max(discounted_strike - discounted_spot, 0.0)
            upper_bound = discounted_strike

        return lower_bound, upper_bound
