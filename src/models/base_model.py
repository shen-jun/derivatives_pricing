"""
This is my base class definition for every option pricing model in this project.

In this project, I designed every single pricing model (Black-Scholes-Merton,
Black-76, CRR/LR binomial trees, Monte Carlo, and the Finite Difference PDE
solver) to inherit from the same abstract base class, OptionModel. My reasoning
is that a risk desk needs to be able to swap the pricing model underneath a
given option contract without changing the way the rest of the codebase (the
Greeks engine, the scenario engine, the dashboard) talks to it. As long as a
model exposes price(), delta(), gamma(), vega(), theta() and rho(), everything
else in my platform can treat it as an interchangeable pricing "plug-in".

I store all contract and market parameters (spot, strike, time to expiry,
risk-free rate, volatility, dividend yield, option type) directly on the
instance in __init__, so every subclass constructor looks and behaves the
same way from the caller's point of view.
"""

from abc import ABC, abstractmethod


class OptionModel(ABC):
    """
    My abstract base class for all option pricing models.

    Parameters
    ----------
    spot : float
        Current price of the underlying asset (or futures price for Black-76).
    strike : float
        Strike price of the option contract.
    time_to_expiry : float
        Time to expiry expressed in years (e.g. 0.5 for six months).
    risk_free_rate : float
        Continuously compounded risk-free rate, e.g. 0.05 for 5%.
    volatility : float
        Annualized volatility of the underlying, e.g. 0.20 for 20%.
    dividend_yield : float, optional
        Continuous dividend yield paid by the underlying. Defaults to 0.0.
    option_type : str, optional
        Either "call" or "put". Defaults to "call".
    """

    def __init__(
        self,
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield=0.0,
        option_type="call",
    ):
        if spot <= 0:
            raise ValueError("I require spot price to be strictly positive.")
        if strike <= 0:
            raise ValueError("I require strike price to be strictly positive.")
        if time_to_expiry < 0:
            raise ValueError("I require time to expiry to be non-negative.")
        if volatility < 0:
            raise ValueError("I require volatility to be non-negative.")

        option_type = option_type.lower().strip()
        if option_type not in ("call", "put"):
            raise ValueError("I only accept option_type of 'call' or 'put'.")

        self.spot = float(spot)
        self.strike = float(strike)
        self.time_to_expiry = float(time_to_expiry)
        self.risk_free_rate = float(risk_free_rate)
        self.volatility = float(volatility)
        self.dividend_yield = float(dividend_yield)
        self.option_type = option_type

    @abstractmethod
    def price(self):
        """I require every subclass to return the fair value of the option."""
        raise NotImplementedError

    @abstractmethod
    def delta(self):
        """I require every subclass to return sensitivity of price to spot."""
        raise NotImplementedError

    @abstractmethod
    def gamma(self):
        """I require every subclass to return sensitivity of delta to spot."""
        raise NotImplementedError

    @abstractmethod
    def vega(self):
        """I require every subclass to return sensitivity of price to volatility."""
        raise NotImplementedError

    @abstractmethod
    def theta(self):
        """I require every subclass to return sensitivity of price to time decay."""
        raise NotImplementedError

    @abstractmethod
    def rho(self):
        """I require every subclass to return sensitivity of price to interest rate."""
        raise NotImplementedError

    def greeks_summary(self):
        """
        I use this convenience method across my Greeks engine and dashboard
        so I do not have to call each Greek individually every time I want a
        full risk snapshot of a single option position.
        """
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(),
            "rho": self.rho(),
        }

    def intrinsic_value(self):
        """
        I compute the intrinsic value of the option, which I reuse in several
        places: early exercise checks in my binomial trees, payoff diagrams,
        and no-arbitrage bound checks in my validation tests.
        """
        if self.option_type == "call":
            return max(self.spot - self.strike, 0.0)
        return max(self.strike - self.spot, 0.0)
