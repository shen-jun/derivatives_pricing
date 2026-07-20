"""
This is my implementation of the Black-Scholes-Merton model for pricing
European call and put options on a dividend-paying underlying.

I use this model for the "European Call & Put" bucket of my option-type to
pricing-model mapping. Every formula below is derived in Section 2 of my
math_and_logic.tex document, so I only leave short comments here pointing
back to the relevant equation instead of re-deriving everything in code
comments.
"""

import math

from src.models.base_model import OptionModel
from src.models.math_utils import normal_cdf, normal_pdf, d1_d2


class BlackScholesModel(OptionModel):
    """
    My closed-form pricer for European options under the Black-Scholes-Merton
    framework, including continuous dividend yield support.
    """

    def _compute_d1_d2(self):
        return d1_d2(
            self.spot,
            self.strike,
            self.time_to_expiry,
            self.risk_free_rate,
            self.volatility,
            self.dividend_yield,
        )

    def price(self):
        """
        I return the fair value of the option under Black-Scholes-Merton.

        Call = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
        Put  = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)
        """
        if self.time_to_expiry == 0:
            return self.intrinsic_value()

        d1, d2 = self._compute_d1_d2()
        discounted_spot = self.spot * math.exp(-self.dividend_yield * self.time_to_expiry)
        discounted_strike = self.strike * math.exp(-self.risk_free_rate * self.time_to_expiry)

        if self.option_type == "call":
            return discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
        else:
            return discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1)

    def delta(self):
        """
        Delta_call =  exp(-q*T) * N(d1)
        Delta_put  = -exp(-q*T) * N(-d1)
        """
        if self.time_to_expiry == 0:
            if self.option_type == "call":
                return 1.0 if self.spot > self.strike else 0.0
            return -1.0 if self.spot < self.strike else 0.0

        d1, _ = self._compute_d1_d2()
        discount_factor_dividend = math.exp(-self.dividend_yield * self.time_to_expiry)

        if self.option_type == "call":
            return discount_factor_dividend * normal_cdf(d1)
        else:
            return -discount_factor_dividend * normal_cdf(-d1)

    def gamma(self):
        """
        Gamma is identical for calls and puts:

        Gamma = exp(-q*T) * n(d1) / (S * sigma * sqrt(T))
        """
        if self.time_to_expiry == 0:
            return 0.0

        d1, _ = self._compute_d1_d2()
        discount_factor_dividend = math.exp(-self.dividend_yield * self.time_to_expiry)
        return (
            discount_factor_dividend
            * normal_pdf(d1)
            / (self.spot * self.volatility * math.sqrt(self.time_to_expiry))
        )

    def vega(self):
        """
        Vega is identical for calls and puts, expressed here per unit change
        in volatility (e.g. multiply by 0.01 outside this method if I want
        the change per 1 vol point):

        Vega = S * exp(-q*T) * n(d1) * sqrt(T)
        """
        if self.time_to_expiry == 0:
            return 0.0

        d1, _ = self._compute_d1_d2()
        discount_factor_dividend = math.exp(-self.dividend_yield * self.time_to_expiry)
        return self.spot * discount_factor_dividend * normal_pdf(d1) * math.sqrt(self.time_to_expiry)

    def theta(self):
        """
        I return Theta expressed as the change in option value per year
        (I divide by 365 in my reporting layer whenever I want per-day Theta).

        Theta_call = -S*exp(-q*T)*n(d1)*sigma / (2*sqrt(T))
                     - r*K*exp(-r*T)*N(d2)
                     + q*S*exp(-q*T)*N(d1)

        Theta_put  = -S*exp(-q*T)*n(d1)*sigma / (2*sqrt(T))
                     + r*K*exp(-r*T)*N(-d2)
                     - q*S*exp(-q*T)*N(-d1)
        """
        if self.time_to_expiry == 0:
            return 0.0

        d1, d2 = self._compute_d1_d2()
        discount_factor_dividend = math.exp(-self.dividend_yield * self.time_to_expiry)
        discount_factor_rate = math.exp(-self.risk_free_rate * self.time_to_expiry)

        time_decay_term = -(
            self.spot * discount_factor_dividend * normal_pdf(d1) * self.volatility
        ) / (2.0 * math.sqrt(self.time_to_expiry))

        if self.option_type == "call":
            rate_term = -self.risk_free_rate * self.strike * discount_factor_rate * normal_cdf(d2)
            dividend_term = self.dividend_yield * self.spot * discount_factor_dividend * normal_cdf(d1)
            return time_decay_term + rate_term + dividend_term
        else:
            rate_term = self.risk_free_rate * self.strike * discount_factor_rate * normal_cdf(-d2)
            dividend_term = -self.dividend_yield * self.spot * discount_factor_dividend * normal_cdf(-d1)
            return time_decay_term + rate_term + dividend_term

    def rho(self):
        """
        Rho_call =  K * T * exp(-r*T) * N(d2)
        Rho_put  = -K * T * exp(-r*T) * N(-d2)

        I express this per unit change in the risk-free rate (multiply by
        0.0001 outside this method if I want the change per 1 basis point).
        """
        if self.time_to_expiry == 0:
            return 0.0

        _, d2 = self._compute_d1_d2()
        discount_factor_rate = math.exp(-self.risk_free_rate * self.time_to_expiry)

        if self.option_type == "call":
            return self.strike * self.time_to_expiry * discount_factor_rate * normal_cdf(d2)
        else:
            return -self.strike * self.time_to_expiry * discount_factor_rate * normal_cdf(-d2)
