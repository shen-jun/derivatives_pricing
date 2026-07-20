"""
This is my implementation of the Black-76 model, which I use for the
"Options on Futures" bucket of my option-type to pricing-model mapping.

The key difference versus my Black-Scholes-Merton class is that here I treat
`self.spot` as the futures/forward price F rather than a spot price. Because
a futures contract has no carry cost (it is already a forward price), I do
not apply a separate dividend yield term the way I do in black_scholes.py.
I still keep the dividend_yield attribute on the base class for interface
consistency, but I never use it here and I document that clearly below.
"""

import math

from src.models.base_model import OptionModel
from src.models.math_utils import normal_cdf, normal_pdf


class Black76Model(OptionModel):
    """
    My closed-form pricer for European options on futures/forwards.

    I reuse `self.spot` to store the futures price F. I ignore
    `self.dividend_yield` entirely in this model because the futures price
    already embeds the cost of carry.
    """

    def _compute_d1_d2(self):
        futures_price = self.spot
        numerator = math.log(futures_price / self.strike) + 0.5 * self.volatility * self.volatility * self.time_to_expiry
        denominator = self.volatility * math.sqrt(self.time_to_expiry)
        d1 = numerator / denominator
        d2 = d1 - denominator
        return d1, d2

    def price(self):
        """
        Call = exp(-r*T) * [F * N(d1) - K * N(d2)]
        Put  = exp(-r*T) * [K * N(-d2) - F * N(-d1)]
        """
        if self.time_to_expiry == 0:
            return self.intrinsic_value()

        futures_price = self.spot
        d1, d2 = self._compute_d1_d2()
        discount_factor = math.exp(-self.risk_free_rate * self.time_to_expiry)

        if self.option_type == "call":
            return discount_factor * (futures_price * normal_cdf(d1) - self.strike * normal_cdf(d2))
        else:
            return discount_factor * (self.strike * normal_cdf(-d2) - futures_price * normal_cdf(-d1))

    def delta(self):
        """
        Delta_call =  exp(-r*T) * N(d1)
        Delta_put  = -exp(-r*T) * N(-d1)

        I express this as the sensitivity of the option value to the futures
        price, holding the discount factor fixed.
        """
        if self.time_to_expiry == 0:
            if self.option_type == "call":
                return 1.0 if self.spot > self.strike else 0.0
            return -1.0 if self.spot < self.strike else 0.0

        d1, _ = self._compute_d1_d2()
        discount_factor = math.exp(-self.risk_free_rate * self.time_to_expiry)

        if self.option_type == "call":
            return discount_factor * normal_cdf(d1)
        else:
            return -discount_factor * normal_cdf(-d1)

    def gamma(self):
        """
        Gamma = exp(-r*T) * n(d1) / (F * sigma * sqrt(T))
        """
        if self.time_to_expiry == 0:
            return 0.0

        futures_price = self.spot
        d1, _ = self._compute_d1_d2()
        discount_factor = math.exp(-self.risk_free_rate * self.time_to_expiry)
        return discount_factor * normal_pdf(d1) / (futures_price * self.volatility * math.sqrt(self.time_to_expiry))

    def vega(self):
        """
        Vega = F * exp(-r*T) * n(d1) * sqrt(T)
        """
        if self.time_to_expiry == 0:
            return 0.0

        futures_price = self.spot
        d1, _ = self._compute_d1_d2()
        discount_factor = math.exp(-self.risk_free_rate * self.time_to_expiry)
        return futures_price * discount_factor * normal_pdf(d1) * math.sqrt(self.time_to_expiry)

    def theta(self):
        """
        Theta_call = -F*exp(-r*T)*n(d1)*sigma / (2*sqrt(T))
                     + r*F*exp(-r*T)*N(d1)
                     - r*K*exp(-r*T)*N(d2)

        Theta_put  = -F*exp(-r*T)*n(d1)*sigma / (2*sqrt(T))
                     - r*F*exp(-r*T)*N(-d1)
                     + r*K*exp(-r*T)*N(-d2)

        Both terms come from differentiating the discounted expectation with
        respect to time, keeping in mind that the discount factor itself
        also decays with time.
        """
        if self.time_to_expiry == 0:
            return 0.0

        futures_price = self.spot
        d1, d2 = self._compute_d1_d2()
        discount_factor = math.exp(-self.risk_free_rate * self.time_to_expiry)

        time_decay_term = -(futures_price * discount_factor * normal_pdf(d1) * self.volatility) / (
            2.0 * math.sqrt(self.time_to_expiry)
        )

        if self.option_type == "call":
            carry_term = self.risk_free_rate * futures_price * discount_factor * normal_cdf(d1)
            rate_term = -self.risk_free_rate * self.strike * discount_factor * normal_cdf(d2)
            return time_decay_term + carry_term + rate_term
        else:
            carry_term = -self.risk_free_rate * futures_price * discount_factor * normal_cdf(-d1)
            rate_term = self.risk_free_rate * self.strike * discount_factor * normal_cdf(-d2)
            return time_decay_term + carry_term + rate_term

    def rho(self):
        """
        Because the only place the risk-free rate enters the Black-76
        formula is through the discount factor exp(-r*T) applied to the
        whole expectation, I get a very clean result:

        Rho = -T * Price

        I derive this explicitly in math_and_logic.tex rather than just
        asserting it, since it is a slightly unusual shortcut compared to
        Black-Scholes-Merton Rho.
        """
        if self.time_to_expiry == 0:
            return 0.0

        return -self.time_to_expiry * self.price()
