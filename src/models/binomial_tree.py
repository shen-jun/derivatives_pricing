"""
This is my implementation of the two tree-based models I use for the
"American Call & Put" bucket of my option-type to pricing-model mapping:
the Cox-Ross-Rubinstein (CRR) binomial tree and the Leisen-Reimer (LR)
binomial tree.

I built a shared base class, BinomialTreeModel, that owns the parts both
trees have in common: the backward induction loop, the early exercise
check, and the Greeks extraction logic (delta/gamma/theta straight from the
tree nodes, vega/rho by bump-and-reprice). CRRBinomialTree and
LeisenReimerTree only differ in how they compute the up-move factor u,
down-move factor d and risk-neutral probability p, so I isolated that piece
into a single method, `_tree_parameters()`, that each subclass overrides.

I intentionally build the tree with plain nested Python lists and explicit
for-loops instead of a vectorized numpy array, because the whole point of
this file is to make the backward induction and early-exercise logic
readable step by step.
"""

import math

from src.models.base_model import OptionModel


class BinomialTreeModel(OptionModel):
    """
    My shared base class for tree-based option pricing models.

    Parameters
    ----------
    steps : int
        Number of discrete time steps in the tree. More steps means a
        better approximation of the continuous-time price, at the cost of
        more computation (backward induction is O(steps^2)).
    american : bool
        If True, I check for early exercise at every node. If False, I
        collapse the tree into a European-style pricer, which I use in my
        model validation tests to confirm American price >= European price.
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
        steps=200,
        american=True,
    ):
        super().__init__(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type)
        if steps < 2:
            raise ValueError("I require at least 2 steps to extract delta, gamma and theta from the tree.")
        self.steps = steps
        self.american = american

    def _tree_parameters(self):
        """
        Every subclass must return a tuple (dt, u, d, p) where:
        dt = length of one time step in years
        u  = up-move multiplicative factor
        d  = down-move multiplicative factor
        p  = risk-neutral probability of an up move
        """
        raise NotImplementedError("Subclasses of BinomialTreeModel must implement _tree_parameters().")

    def _payoff(self, asset_price):
        """I compute the option payoff at a given asset price node."""
        if self.option_type == "call":
            return max(asset_price - self.strike, 0.0)
        return max(self.strike - asset_price, 0.0)

    def _build_full_grid(self):
        """
        I run the complete backward induction and return the full grid of
        option values, indexed as grid[step][node], where node counts the
        number of up-moves that occurred by that step (0 to step).

        I keep the full grid (rather than just the root value) because I
        reuse the early steps of the grid to compute delta, gamma and theta
        directly from neighboring tree nodes, following the standard
        approach described in Hull's "Options, Futures, and Other
        Derivatives".
        """
        dt, u, d, p = self._tree_parameters()
        discount = math.exp(-self.risk_free_rate * dt)
        n = self.steps

        if p <= 0.0 or p >= 1.0:
            raise ValueError(
                "I detected a risk-neutral probability outside (0, 1), which means my tree "
                "parameters admit arbitrage. Try increasing the number of steps."
            )

        # I allocate the grid as a list of lists. grid[i] holds i+1 values.
        grid = [[0.0] * (i + 1) for i in range(n + 1)]

        # Step 1: terminal payoffs at maturity (step n).
        for node_index in range(n + 1):
            asset_price = self.spot * (u ** node_index) * (d ** (n - node_index))
            grid[n][node_index] = self._payoff(asset_price)

        # Step 2: backward induction from maturity down to today.
        for step in range(n - 1, -1, -1):
            for node_index in range(step + 1):
                continuation_value = discount * (
                    p * grid[step + 1][node_index + 1] + (1.0 - p) * grid[step + 1][node_index]
                )
                if self.american:
                    asset_price = self.spot * (u ** node_index) * (d ** (step - node_index))
                    intrinsic_value = self._payoff(asset_price)
                    grid[step][node_index] = max(continuation_value, intrinsic_value)
                else:
                    grid[step][node_index] = continuation_value

        return grid, u, d, dt

    def price(self):
        grid, _, _, _ = self._build_full_grid()
        return grid[0][0]

    def delta(self):
        """
        I estimate delta from the two nodes at step 1 of the tree:

        Delta = [V(up) - V(down)] / [S*u - S*d]
        """
        grid, u, d, _ = self._build_full_grid()
        asset_price_up = self.spot * u
        asset_price_down = self.spot * d
        return (grid[1][1] - grid[1][0]) / (asset_price_up - asset_price_down)

    def gamma(self):
        """
        I estimate gamma from the three nodes at step 2 of the tree, using
        two local deltas and differencing them:

        Delta_upper = [V(uu) - V(ud)] / [S*u^2 - S*u*d]
        Delta_lower = [V(ud) - V(dd)] / [S*u*d - S*d^2]
        Gamma       = [Delta_upper - Delta_lower] / [0.5*(S*u^2 - S*d^2)]
        """
        grid, u, d, _ = self._build_full_grid()
        asset_price_uu = self.spot * u * u
        asset_price_ud = self.spot * u * d
        asset_price_dd = self.spot * d * d

        delta_upper = (grid[2][2] - grid[2][1]) / (asset_price_uu - asset_price_ud)
        delta_lower = (grid[2][1] - grid[2][0]) / (asset_price_ud - asset_price_dd)

        return (delta_upper - delta_lower) / (0.5 * (asset_price_uu - asset_price_dd))

    def theta(self):
        """
        I estimate theta using the "middle" node at step 2, which shares the
        same asset price as the root node (S*u*d = S), so I get a clean
        central-difference-in-time estimate:

        Theta = [V(step 2, middle node) - V(step 0)] / (2 * dt)
        """
        grid, _, _, dt = self._build_full_grid()
        return (grid[2][1] - grid[0][0]) / (2.0 * dt)

    def early_exercise_boundary(self):
        """
        I scan my full backward-induction grid and, for every time step
        before maturity, find the asset price closest to the strike at
        which the option value equals its intrinsic value (immediate
        exercise is optimal). Plotting this critical price against time
        traces out the early exercise boundary, which I use in my
        "Early Exercise Boundary" chart.

        I only run this for American options, since a European tree never
        exercises early and has no boundary to trace.
        """
        if not self.american:
            raise ValueError("I can only compute an early exercise boundary for an American option.")

        grid, u, d, dt = self._build_full_grid()
        boundary_points = []

        for step in range(self.steps):
            critical_price = None
            for node_index in range(step + 1):
                asset_price = self.spot * (u ** node_index) * (d ** (step - node_index))
                intrinsic_value = self._payoff(asset_price)
                is_exercise_optimal = intrinsic_value > 0 and abs(grid[step][node_index] - intrinsic_value) < 1e-6

                if is_exercise_optimal:
                    if critical_price is None:
                        critical_price = asset_price
                    elif self.option_type == "call" and asset_price < critical_price:
                        critical_price = asset_price
                    elif self.option_type == "put" and asset_price > critical_price:
                        critical_price = asset_price

            boundary_points.append((step * dt, critical_price))

        return boundary_points

    def _rebuild_with(self, **overrides):
        """
        I use this helper to create a fresh copy of the current tree with
        one or more parameters bumped, which is how I estimate vega and rho
        by finite differences (bump-and-reprice). Trees do not give me a
        clean analytic vega/rho the way delta/gamma/theta come naturally
        out of the node structure, so bump-and-reprice is the standard
        industry approach here.
        """
        params = {
            "spot": self.spot,
            "strike": self.strike,
            "time_to_expiry": self.time_to_expiry,
            "risk_free_rate": self.risk_free_rate,
            "volatility": self.volatility,
            "dividend_yield": self.dividend_yield,
            "option_type": self.option_type,
            "steps": self.steps,
            "american": self.american,
        }
        params.update(overrides)
        return type(self)(**params)

    def vega(self):
        """
        I bump volatility up and down by a small amount and take a central
        difference of the resulting prices.
        """
        bump = 0.0001
        model_up = self._rebuild_with(volatility=self.volatility + bump)
        model_down = self._rebuild_with(volatility=max(self.volatility - bump, 1e-6))
        return (model_up.price() - model_down.price()) / (2.0 * bump)

    def rho(self):
        """
        I bump the risk-free rate up and down by a small amount and take a
        central difference of the resulting prices.
        """
        bump = 0.0001
        model_up = self._rebuild_with(risk_free_rate=self.risk_free_rate + bump)
        model_down = self._rebuild_with(risk_free_rate=self.risk_free_rate - bump)
        return (model_up.price() - model_down.price()) / (2.0 * bump)


class CRRBinomialTree(BinomialTreeModel):
    """
    My Cox-Ross-Rubinstein (1979) implementation.

    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = [exp((r - q) * dt) - d] / (u - d)
    """

    def _tree_parameters(self):
        dt = self.time_to_expiry / self.steps
        u = math.exp(self.volatility * math.sqrt(dt))
        d = 1.0 / u
        growth_factor = math.exp((self.risk_free_rate - self.dividend_yield) * dt)
        p = (growth_factor - d) / (u - d)
        return dt, u, d, p


class LeisenReimerTree(BinomialTreeModel):
    """
    My Leisen-Reimer (1996) implementation. The LR tree is constructed so
    that it converges to the Black-Scholes-Merton price smoothly (without
    the oscillation CRR shows as the number of steps changes), by matching
    the tree's risk-neutral probability directly to the Black-Scholes d1/d2
    terms through the Peizer-Pratt inversion formula.

    I force the number of steps to be odd, since the LR construction
    requires this for the inversion formula to line up with the strike.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.steps % 2 == 0:
            self.steps += 1

    @staticmethod
    def _peizer_pratt_inversion(z, n):
        """
        h(z) = 0.5 + sign(z) * sqrt( 0.25 - 0.25 * exp( -(z / (n + 1/3 + 0.1/(n+1)))^2 * (n + 1/6) ) )
        """
        sign_z = 1.0 if z >= 0 else -1.0
        denominator_term = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
        exponent = -((z / denominator_term) ** 2) * (n + 1.0 / 6.0)
        inside_sqrt = 0.25 - 0.25 * math.exp(exponent)
        # Guard against tiny negative numbers from floating point noise.
        inside_sqrt = max(inside_sqrt, 0.0)
        return 0.5 + sign_z * math.sqrt(inside_sqrt)

    def _tree_parameters(self):
        dt = self.time_to_expiry / self.steps
        n = float(self.steps)

        # I need d1 and d2 from the standard Black-Scholes-Merton formulas,
        # computed once at inception (using the full time to expiry, not dt).
        numerator = math.log(self.spot / self.strike) + (
            self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility * self.volatility
        ) * self.time_to_expiry
        denominator = self.volatility * math.sqrt(self.time_to_expiry)
        d1 = numerator / denominator
        d2 = d1 - denominator

        p = self._peizer_pratt_inversion(d2, n)
        p_prime = self._peizer_pratt_inversion(d1, n)

        growth_factor = math.exp((self.risk_free_rate - self.dividend_yield) * dt)
        u = growth_factor * p_prime / p
        d = (growth_factor - p * u) / (1.0 - p)

        return dt, u, d, p
