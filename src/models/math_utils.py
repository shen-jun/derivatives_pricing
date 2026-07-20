"""
I put my small shared math helpers here so I do not repeat the same normal
distribution formulas across black_scholes.py, black76.py and the calibration
module. I deliberately wrote these by hand with the math module instead of
importing scipy.stats.norm everywhere, because I want the reader of this
project to see exactly which closed-form formula is being evaluated at each
step instead of hiding it behind a library call.
"""

import math


def normal_cdf(x):
    """
    I compute the standard normal cumulative distribution function N(x)
    using the error function identity:

        N(x) = 0.5 * (1 + erf(x / sqrt(2)))

    This is the exact same value that scipy.stats.norm.cdf(x) would give me,
    but written explicitly so the formula is visible in the code.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x):
    """
    I compute the standard normal probability density function n(x):

        n(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
    """
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0.0):
    """
    I compute the d1 and d2 terms that show up in every Black-Scholes-style
    closed-form formula I use in this project (Black-Scholes-Merton and
    Black-76 both reduce to this same structure once I treat the futures
    price as a spot price with a zero cost of carry).

        d1 = [ln(S/K) + (r - q + 0.5*sigma^2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
    """
    if time_to_expiry <= 0 or volatility <= 0:
        raise ValueError(
            "I cannot compute d1/d2 when time_to_expiry or volatility is zero. "
            "Handle the expiry/zero-vol edge case before calling this function."
        )

    numerator = math.log(spot / strike) + (
        risk_free_rate - dividend_yield + 0.5 * volatility * volatility
    ) * time_to_expiry
    denominator = volatility * math.sqrt(time_to_expiry)

    d1 = numerator / denominator
    d2 = d1 - denominator
    return d1, d2
