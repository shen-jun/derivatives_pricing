"""
This is my Finite Difference PDE solver, which I use for the "Barrier
Options" bucket of my option-type to pricing-model mapping.

I solve the Black-Scholes-Merton partial differential equation directly on
a discretized (price, time) grid using the explicit finite difference
method, which is the most transparent of the standard PDE schemes (as
opposed to implicit or Crank-Nicolson, which need a tridiagonal solver). I
chose the explicit scheme specifically because I want the update formula at
each grid node to be a single readable line, at the cost of needing a
finer time grid to remain numerically stable.

I price knock-out barriers (down-and-out, up-and-out) directly on the PDE
grid by forcing the option value to zero at any node beyond the barrier, at
every time step. I price knock-in barriers using the standard market
in/out parity relationship:

    knock-in price = vanilla price - knock-out price

which holds because holding both a knock-in and the corresponding knock-out
barrier option is equivalent to holding the vanilla option (exactly one of
the two ever pays off, depending on whether the barrier is breached).
"""

import math

from src.models.base_model import OptionModel
from src.models.black_scholes import BlackScholesModel


class FiniteDifferencePDE(OptionModel):
    """
    My explicit finite difference pricer for single-barrier options.

    Parameters
    ----------
    barrier_level : float
        The barrier price level.
    barrier_type : str
        One of "down-and-out", "up-and-out", "down-and-in", "up-and-in".
    num_price_steps : int
        Number of intervals in the underlying price dimension of my grid.
    num_time_steps : int
        Minimum number of intervals in the time dimension of my grid. I may
        internally increase this if it is too small to keep the explicit
        scheme numerically stable.
    spot_max_multiplier : float
        I set the top of my price grid to this multiple of the largest of
        spot, strike and barrier, so the grid comfortably spans the region
        the option actually cares about.
    """

    VALID_BARRIER_TYPES = ("down-and-out", "up-and-out", "down-and-in", "up-and-in")

    def __init__(
        self,
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield=0.0,
        option_type="call",
        barrier_level=None,
        barrier_type="down-and-out",
        num_price_steps=100,
        num_time_steps=1000,
        spot_max_multiplier=3.0,
    ):
        super().__init__(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type)

        if barrier_level is None or barrier_level <= 0:
            raise ValueError("I require a strictly positive barrier_level for a barrier option.")
        if barrier_type not in self.VALID_BARRIER_TYPES:
            raise ValueError(f"I only accept barrier_type in {self.VALID_BARRIER_TYPES}.")

        self.barrier_level = float(barrier_level)
        self.barrier_type = barrier_type
        self.num_price_steps = num_price_steps
        self.num_time_steps = num_time_steps
        self.spot_max = spot_max_multiplier * max(self.spot, self.strike, self.barrier_level)

    def _stable_num_time_steps(self):
        """
        I use a conservative sufficient condition for stability of the
        explicit finite difference scheme applied to the Black-Scholes PDE:

            dt <= 1 / (sigma^2 * M^2 + r)

        where M is the number of price steps. If the num_time_steps I was
        given by the caller does not satisfy this bound, I silently round
        it up so the scheme stays stable, rather than returning numerical
        garbage.
        """
        max_index = self.num_price_steps
        dt_bound = 1.0 / (self.volatility * self.volatility * max_index * max_index + self.risk_free_rate)
        required_steps = int(math.ceil(self.time_to_expiry / dt_bound)) + 1
        return max(self.num_time_steps, required_steps)

    def _terminal_payoff(self, asset_price):
        if self.option_type == "call":
            return max(asset_price - self.strike, 0.0)
        return max(self.strike - asset_price, 0.0)

    def _is_knocked_out(self, asset_price):
        """I check whether a given grid price has crossed the barrier for a knock-out contract."""
        if self.barrier_type in ("down-and-out", "down-and-in"):
            return asset_price <= self.barrier_level
        else:  # up-and-out, up-and-in
            return asset_price >= self.barrier_level

    def _price_knock_out_grid(self):
        """
        I build the full explicit finite difference grid for the knock-out
        version of this contract and return the full 2D grid together with
        the price step size dS and time step size dt, so that price(),
        delta(), gamma() and theta() can all reuse the same grid without
        recomputation.

        grid[i][j] = option value at price node i (S_i = i * dS) and time
        node j, where j = 0 is today and j = num_time_steps is maturity.
        """
        num_time_steps = self._stable_num_time_steps()
        num_price_steps = self.num_price_steps

        dS = self.spot_max / num_price_steps
        dt = self.time_to_expiry / num_time_steps

        grid = [[0.0] * (num_time_steps + 1) for _ in range(num_price_steps + 1)]

        # Terminal condition at maturity (time index = num_time_steps).
        for i in range(num_price_steps + 1):
            asset_price = i * dS
            payoff = self._terminal_payoff(asset_price)
            if self._is_knocked_out(asset_price):
                payoff = 0.0
            grid[i][num_time_steps] = payoff

        # Backward induction from maturity (j = num_time_steps) down to today (j = 0).
        for j in range(num_time_steps - 1, -1, -1):
            time_remaining = self.time_to_expiry - j * dt

            for i in range(1, num_price_steps):
                asset_price = i * dS
                a_i = 0.5 * dt * (self.volatility ** 2 * i * i - (self.risk_free_rate - self.dividend_yield) * i)
                b_i = 1.0 - dt * (self.volatility ** 2 * i * i + self.risk_free_rate)
                c_i = 0.5 * dt * (self.volatility ** 2 * i * i + (self.risk_free_rate - self.dividend_yield) * i)

                continuation_value = (
                    a_i * grid[i - 1][j + 1] + b_i * grid[i][j + 1] + c_i * grid[i + 1][j + 1]
                )

                if self._is_knocked_out(asset_price):
                    grid[i][j] = 0.0
                else:
                    grid[i][j] = continuation_value

            # Boundary conditions at the edges of my price grid.
            if self.option_type == "call":
                lower_boundary_value = 0.0
                upper_boundary_value = self.spot_max * math.exp(
                    -self.dividend_yield * time_remaining
                ) - self.strike * math.exp(-self.risk_free_rate * time_remaining)
            else:
                lower_boundary_value = self.strike * math.exp(-self.risk_free_rate * time_remaining)
                upper_boundary_value = 0.0

            grid[0][j] = 0.0 if self._is_knocked_out(0.0) else lower_boundary_value
            grid[num_price_steps][j] = 0.0 if self._is_knocked_out(self.spot_max) else upper_boundary_value

        return grid, dS, dt, num_price_steps, num_time_steps

    def _knock_out_price_from_grid(self):
        grid, dS, dt, num_price_steps, num_time_steps = self._price_knock_out_grid()
        spot_index = self.spot / dS

        # I linearly interpolate between the two nearest grid nodes since my
        # spot price will rarely land exactly on a grid point.
        lower_index = int(math.floor(spot_index))
        upper_index = min(lower_index + 1, num_price_steps)
        weight_upper = spot_index - lower_index
        weight_lower = 1.0 - weight_upper

        price_today = weight_lower * grid[lower_index][0] + weight_upper * grid[upper_index][0]
        return price_today, grid, dS, dt, lower_index, upper_index

    def _is_knock_in(self):
        return self.barrier_type in ("down-and-in", "up-and-in")

    def _matching_knock_out_type(self):
        return self.barrier_type.replace("-in", "-out")

    def price(self):
        if self._is_knock_in():
            vanilla_price = BlackScholesModel(
                self.spot, self.strike, self.time_to_expiry, self.risk_free_rate,
                self.volatility, self.dividend_yield, self.option_type,
            ).price()
            knock_out_twin = self._rebuild_with(barrier_type=self._matching_knock_out_type())
            return vanilla_price - knock_out_twin.price()

        price_today, _, _, _, _, _ = self._knock_out_price_from_grid()
        return price_today

    def _rebuild_with(self, **overrides):
        params = {
            "spot": self.spot,
            "strike": self.strike,
            "time_to_expiry": self.time_to_expiry,
            "risk_free_rate": self.risk_free_rate,
            "volatility": self.volatility,
            "dividend_yield": self.dividend_yield,
            "option_type": self.option_type,
            "barrier_level": self.barrier_level,
            "barrier_type": self.barrier_type,
            "num_price_steps": self.num_price_steps,
            "num_time_steps": self.num_time_steps,
            "spot_max_multiplier": self.spot_max / max(self.spot, self.strike, self.barrier_level),
        }
        params.update(overrides)
        return FiniteDifferencePDE(**params)

    def delta(self):
        if self._is_knock_in():
            return self._greek_by_bump("delta")

        _, grid, dS, dt, lower_index, upper_index = self._knock_out_price_from_grid()
        if lower_index < 1 or upper_index > self.num_price_steps - 1:
            return self._greek_by_bump("delta")

        value_up = grid[lower_index + 1][0]
        value_down = grid[lower_index - 1][0]
        return (value_up - value_down) / (2.0 * dS)

    def gamma(self):
        if self._is_knock_in():
            return self._greek_by_bump("gamma")

        _, grid, dS, dt, lower_index, upper_index = self._knock_out_price_from_grid()
        if lower_index < 1 or upper_index > self.num_price_steps - 1:
            return self._greek_by_bump("gamma")

        value_up = grid[lower_index + 1][0]
        value_mid = grid[lower_index][0]
        value_down = grid[lower_index - 1][0]
        return (value_up - 2.0 * value_mid + value_down) / (dS * dS)

    def theta(self):
        if self._is_knock_in():
            return self._greek_by_bump("theta")

        _, grid, dS, dt, lower_index, upper_index = self._knock_out_price_from_grid()
        value_today = grid[lower_index][0]
        value_next_period = grid[lower_index][1]
        return (value_next_period - value_today) / dt

    def _greek_by_bump(self, greek_name):
        """
        I fall back to a simple bump-and-reprice finite difference whenever
        the spot price sits too close to the edge of my grid to safely read
        off a neighboring node (or for knock-in contracts, where I compute
        the Greek from the in/out parity relationship applied to bumped
        prices rather than differentiating the grid directly).
        """
        if greek_name == "delta":
            bump = self.spot * 0.01
            model_up = self._rebuild_with(spot=self.spot + bump)
            model_down = self._rebuild_with(spot=self.spot - bump)
            return (model_up.price() - model_down.price()) / (2.0 * bump)
        elif greek_name == "gamma":
            bump = self.spot * 0.01
            model_up = self._rebuild_with(spot=self.spot + bump)
            model_down = self._rebuild_with(spot=self.spot - bump)
            model_mid = self._rebuild_with()
            return (model_up.price() - 2.0 * model_mid.price() + model_down.price()) / (bump ** 2)
        elif greek_name == "theta":
            bump = 1.0 / 365.0
            shorter_expiry = max(self.time_to_expiry - bump, 1e-6)
            model_shorter = self._rebuild_with(time_to_expiry=shorter_expiry)
            model_base = self._rebuild_with()
            return (model_shorter.price() - model_base.price()) / bump
        raise ValueError(f"I do not know how to bump for greek '{greek_name}'.")

    def vega(self):
        """I bump volatility and reprice the full grid, since vega has no clean analytic form on this grid."""
        bump = 0.01
        model_up = self._rebuild_with(volatility=self.volatility + bump)
        model_down = self._rebuild_with(volatility=max(self.volatility - bump, 1e-6))
        return (model_up.price() - model_down.price()) / (2.0 * bump)

    def rho(self):
        """I bump the risk-free rate and reprice the full grid."""
        bump = 0.0001
        model_up = self._rebuild_with(risk_free_rate=self.risk_free_rate + bump)
        model_down = self._rebuild_with(risk_free_rate=self.risk_free_rate - bump)
        return (model_up.price() - model_down.price()) / (2.0 * bump)

    def barrier_sensitivity(self, barrier_bump=0.01):
        """
        I bump the barrier level up by `barrier_bump` (as a fraction of the
        current barrier) and return the change in option value. This is
        the metric I report in my "Barrier Sensitivity" and "Option Value
        vs Barrier Level" charts.
        """
        base_price = self.price()
        bumped_barrier = self.barrier_level * (1.0 + barrier_bump)
        bumped_model = self._rebuild_with(barrier_level=bumped_barrier)
        return bumped_model.price() - base_price
