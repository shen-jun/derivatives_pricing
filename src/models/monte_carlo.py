"""
This is my Monte Carlo engine, which I use for the three path-dependent
product buckets in my option-type to pricing-model mapping: Asian options,
Lookback options, and Basket options.

I deliberately wrote the path simulation with plain Python for-loops instead
of a vectorized numpy implementation. My reasoning is that a Monte Carlo
engine is exactly the piece of a pricing library where it is easy to hide a
subtle mistake inside a clever vectorized expression, and I want anyone
reading this code (including myself in six months) to be able to follow the
exact sequence of GBM steps that produced each simulated path.

I use common random numbers (the same random_seed) when I bump a parameter
to estimate a Greek by finite differences. This is standard practice for
Monte Carlo Greeks: it removes most of the simulation noise from the
difference between the bumped and unbumped price, which is what makes
finite-difference Greeks usable at all on top of Monte Carlo.
"""

import math
import random

from src.models.base_model import OptionModel


class MonteCarloOptionModel(OptionModel):
    """
    My base class for single-underlying, path-dependent options priced by
    Monte Carlo simulation under the Black-Scholes-Merton GBM dynamics:

        S(t + dt) = S(t) * exp[(r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z]

    where Z is a standard normal random draw.

    Subclasses only need to implement `_payoff_from_path()`, which takes the
    full simulated path (a Python list of prices, including the initial
    spot at index 0) and returns the option's payoff at maturity.
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
        num_paths=10000,
        num_steps=100,
        antithetic=True,
        random_seed=None,
    ):
        super().__init__(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type)
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.antithetic = antithetic
        self.random_seed = random_seed
        self._last_price = None
        self._last_std_error = None

    def _generate_normals(self, rng):
        """I draw one standard normal variate per time step in the path."""
        return [rng.gauss(0.0, 1.0) for _ in range(self.num_steps)]

    def _simulate_path_from_normals(self, normals):
        """
        I roll the GBM recursion forward one time step at a time, using a
        plain for-loop over the list of normal draws I was handed. This is
        the explicit version of the same calculation a vectorized numpy
        cumulative-product would do, just written so every multiplication
        is visible.
        """
        dt = self.time_to_expiry / self.num_steps
        drift = (self.risk_free_rate - self.dividend_yield - 0.5 * self.volatility * self.volatility) * dt
        vol_sqrt_dt = self.volatility * math.sqrt(dt)

        path = [self.spot]
        current_price = self.spot
        for z in normals:
            current_price = current_price * math.exp(drift + vol_sqrt_dt * z)
            path.append(current_price)
        return path

    def _payoff_from_path(self, path):
        raise NotImplementedError("Subclasses of MonteCarloOptionModel must implement _payoff_from_path().")

    def price(self):
        """
        I simulate `num_paths` GBM paths, evaluate the payoff on each one,
        discount the average payoff back to today, and store the standard
        error of my estimate so I can report a confidence interval.

        When antithetic=True, I use the antithetic variates variance
        reduction technique: for every set of normal draws Z I generate, I
        also evaluate the path built from -Z. Averaging a payoff with its
        antithetic partner reduces the variance of my estimator whenever the
        payoff is a monotonic function of the underlying path.
        """
        rng = random.Random(self.random_seed)
        discount_factor = math.exp(-self.risk_free_rate * self.time_to_expiry)
        payoffs = []

        if self.antithetic:
            num_draws = self.num_paths // 2
            for _ in range(num_draws):
                normals = self._generate_normals(rng)
                path_up = self._simulate_path_from_normals(normals)
                antithetic_normals = [-z for z in normals]
                path_down = self._simulate_path_from_normals(antithetic_normals)
                payoffs.append(self._payoff_from_path(path_up))
                payoffs.append(self._payoff_from_path(path_down))
        else:
            for _ in range(self.num_paths):
                normals = self._generate_normals(rng)
                path = self._simulate_path_from_normals(normals)
                payoffs.append(self._payoff_from_path(path))

        num_samples = len(payoffs)
        mean_payoff = sum(payoffs) / num_samples
        discounted_price = discount_factor * mean_payoff

        sum_squared_deviations = 0.0
        for payoff in payoffs:
            sum_squared_deviations += (payoff - mean_payoff) ** 2
        sample_variance = sum_squared_deviations / (num_samples - 1)
        standard_error = discount_factor * math.sqrt(sample_variance / num_samples)

        self._last_price = discounted_price
        self._last_std_error = standard_error
        return discounted_price

    def confidence_interval(self, z_score=1.96):
        """
        I return a (lower_bound, upper_bound) confidence interval around my
        last computed price. The default z_score of 1.96 corresponds to a
        95% confidence level under the normal approximation implied by the
        Central Limit Theorem.
        """
        if self._last_price is None:
            self.price()
        return (
            self._last_price - z_score * self._last_std_error,
            self._last_price + z_score * self._last_std_error,
        )

    def _rebuild_with(self, **overrides):
        """
        I use this to create a bumped copy of the current model for
        finite-difference Greeks. I always keep the same random_seed unless
        it is explicitly overridden, so the bumped and unbumped simulations
        share the same underlying random draws (common random numbers).
        """
        params = {
            "spot": self.spot,
            "strike": self.strike,
            "time_to_expiry": self.time_to_expiry,
            "risk_free_rate": self.risk_free_rate,
            "volatility": self.volatility,
            "dividend_yield": self.dividend_yield,
            "option_type": self.option_type,
            "num_paths": self.num_paths,
            "num_steps": self.num_steps,
            "antithetic": self.antithetic,
            "random_seed": self.random_seed if self.random_seed is not None else 42,
        }
        params.update(overrides)
        return type(self)(**params)

    def delta(self):
        """I bump spot up/down by 1% and take a central finite difference."""
        bump = self.spot * 0.01
        model_up = self._rebuild_with(spot=self.spot + bump)
        model_down = self._rebuild_with(spot=self.spot - bump)
        return (model_up.price() - model_down.price()) / (2.0 * bump)

    def gamma(self):
        """I bump spot up/down by 1% and take a central second difference."""
        bump = self.spot * 0.01
        model_up = self._rebuild_with(spot=self.spot + bump)
        model_down = self._rebuild_with(spot=self.spot - bump)
        model_mid = self._rebuild_with()
        return (model_up.price() - 2.0 * model_mid.price() + model_down.price()) / (bump ** 2)

    def vega(self):
        """I bump volatility up/down by 1 vol point (0.01) and difference."""
        bump = 0.01
        model_up = self._rebuild_with(volatility=self.volatility + bump)
        model_down = self._rebuild_with(volatility=max(self.volatility - bump, 1e-6))
        return (model_up.price() - model_down.price()) / (2.0 * bump)

    def theta(self):
        """
        I bump time to expiry backward by roughly one day and take a
        forward difference, since I cannot simulate past maturity:

        Theta = [V(T - h) - V(T)] / h
        """
        bump = 1.0 / 365.0
        shorter_expiry = max(self.time_to_expiry - bump, 1e-6)
        model_shorter = self._rebuild_with(time_to_expiry=shorter_expiry)
        model_base = self._rebuild_with()
        return (model_shorter.price() - model_base.price()) / bump

    def rho(self):
        """I bump the risk-free rate up/down by 1 basis point and difference."""
        bump = 0.0001
        model_up = self._rebuild_with(risk_free_rate=self.risk_free_rate + bump)
        model_down = self._rebuild_with(risk_free_rate=self.risk_free_rate - bump)
        return (model_up.price() - model_down.price()) / (2.0 * bump)


class AsianOption(MonteCarloOptionModel):
    """
    My arithmetic-average Asian option. The payoff depends on the average
    price observed along the path rather than just the terminal price,
    which is exactly why I cannot use a closed-form Black-Scholes formula
    here and need Monte Carlo instead.

    Call payoff = max(average(S) - K, 0)
    Put payoff  = max(K - average(S), 0)
    """

    def _payoff_from_path(self, path):
        # I average over every simulated observation date, excluding the
        # initial spot at index 0 since that is not a monitoring date.
        monitored_prices = path[1:]
        average_price = sum(monitored_prices) / len(monitored_prices)

        if self.option_type == "call":
            return max(average_price - self.strike, 0.0)
        return max(self.strike - average_price, 0.0)


class LookbackOption(MonteCarloOptionModel):
    """
    My lookback option, supporting both the fixed-strike and floating-strike
    variants.

    Fixed strike:
        Call payoff = max(max(S) - K, 0)
        Put payoff  = max(K - min(S), 0)

    Floating strike (the strike is set to the best/worst observed price):
        Call payoff = S(T) - min(S)
        Put payoff  = max(S) - S(T)
    """

    def __init__(self, *args, floating_strike=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.floating_strike = floating_strike

    def _rebuild_with(self, **overrides):
        model = super()._rebuild_with(**overrides)
        model.floating_strike = self.floating_strike
        return model

    def _payoff_from_path(self, path):
        monitored_prices = path[1:]
        maximum_price = max(monitored_prices)
        minimum_price = min(monitored_prices)
        terminal_price = monitored_prices[-1]

        if self.floating_strike:
            if self.option_type == "call":
                return terminal_price - minimum_price
            return maximum_price - terminal_price

        if self.option_type == "call":
            return max(maximum_price - self.strike, 0.0)
        return max(self.strike - minimum_price, 0.0)


class BasketOption(OptionModel):
    """
    My Monte Carlo pricer for a basket option written on a weighted average
    of several correlated underlyings.

    Because a basket option genuinely needs more than one spot price,
    volatility, and dividend yield, I do not reuse the single-asset
    constructor signature from OptionModel. Instead I store the per-asset
    arrays directly on this class, while still exposing the same
    price()/delta()/gamma()/vega()/theta()/rho() interface as every other
    model in my platform, plus one extra method, correlation_sensitivity(),
    that is specific to basket products.
    """

    def __init__(
        self,
        spots,
        weights,
        volatilities,
        correlation_matrix,
        strike,
        time_to_expiry,
        risk_free_rate,
        dividend_yields=None,
        option_type="call",
        num_paths=5000,
        num_steps=50,
        random_seed=None,
    ):
        if not (len(spots) == len(weights) == len(volatilities)):
            raise ValueError("I require spots, weights and volatilities to have the same length.")

        option_type = option_type.lower().strip()
        if option_type not in ("call", "put"):
            raise ValueError("I only accept option_type of 'call' or 'put'.")

        self.spots = list(spots)
        self.weights = list(weights)
        self.volatilities = list(volatilities)
        self.correlation_matrix = [list(row) for row in correlation_matrix]
        self.strike = float(strike)
        self.time_to_expiry = float(time_to_expiry)
        self.risk_free_rate = float(risk_free_rate)
        self.dividend_yields = list(dividend_yields) if dividend_yields is not None else [0.0] * len(spots)
        self.option_type = option_type
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.random_seed = random_seed
        self.num_assets = len(spots)

        # I keep informational "representative" scalar attributes so this
        # class still displays sensibly anywhere in my platform that expects
        # a generic OptionModel (for example my dashboard summary table).
        self.spot = sum(w * s for w, s in zip(self.weights, self.spots))
        self.volatility = sum(w * v for w, v in zip(self.weights, self.volatilities))

        self._last_price = None
        self._last_std_error = None

    def _cholesky_decomposition(self):
        """
        I compute the lower-triangular Cholesky factor L of my correlation
        matrix by hand with explicit nested loops, so that L @ L^T equals
        the correlation matrix. I use this factor to turn independent
        standard normal draws into correlated ones for my basket
        simulation.
        """
        n = self.num_assets
        lower = [[0.0] * n for _ in range(n)]

        for row in range(n):
            for col in range(row + 1):
                running_total = 0.0
                for k in range(col):
                    running_total += lower[row][k] * lower[col][k]

                if row == col:
                    diagonal_value = self.correlation_matrix[row][row] - running_total
                    lower[row][col] = math.sqrt(max(diagonal_value, 0.0))
                else:
                    if lower[col][col] == 0.0:
                        lower[row][col] = 0.0
                    else:
                        lower[row][col] = (self.correlation_matrix[row][col] - running_total) / lower[col][col]

        return lower

    def _simulate_terminal_prices(self, lower_triangular, rng):
        """
        I advance every asset in the basket forward one time step at a time.
        At each step I draw one independent standard normal per asset, then
        correlate them by multiplying with the Cholesky factor, and finally
        apply the usual GBM update to each asset separately.
        """
        dt = self.time_to_expiry / self.num_steps
        current_prices = list(self.spots)

        for _ in range(self.num_steps):
            independent_normals = [rng.gauss(0.0, 1.0) for _ in range(self.num_assets)]

            correlated_normals = []
            for i in range(self.num_assets):
                running_total = 0.0
                for j in range(i + 1):
                    running_total += lower_triangular[i][j] * independent_normals[j]
                correlated_normals.append(running_total)

            for i in range(self.num_assets):
                drift = (
                    self.risk_free_rate - self.dividend_yields[i] - 0.5 * self.volatilities[i] * self.volatilities[i]
                ) * dt
                diffusion = self.volatilities[i] * math.sqrt(dt) * correlated_normals[i]
                current_prices[i] = current_prices[i] * math.exp(drift + diffusion)

        return current_prices

    def _basket_payoff(self, terminal_prices):
        basket_value = sum(w * p for w, p in zip(self.weights, terminal_prices))
        if self.option_type == "call":
            return max(basket_value - self.strike, 0.0)
        return max(self.strike - basket_value, 0.0)

    def price(self):
        rng = random.Random(self.random_seed)
        lower_triangular = self._cholesky_decomposition()
        discount_factor = math.exp(-self.risk_free_rate * self.time_to_expiry)

        payoffs = []
        for _ in range(self.num_paths):
            terminal_prices = self._simulate_terminal_prices(lower_triangular, rng)
            payoffs.append(self._basket_payoff(terminal_prices))

        num_samples = len(payoffs)
        mean_payoff = sum(payoffs) / num_samples
        price = discount_factor * mean_payoff

        sum_squared_deviations = 0.0
        for payoff in payoffs:
            sum_squared_deviations += (payoff - mean_payoff) ** 2
        sample_variance = sum_squared_deviations / (num_samples - 1)
        self._last_std_error = discount_factor * math.sqrt(sample_variance / num_samples)
        self._last_price = price
        return price

    def confidence_interval(self, z_score=1.96):
        if self._last_price is None:
            self.price()
        return (
            self._last_price - z_score * self._last_std_error,
            self._last_price + z_score * self._last_std_error,
        )

    def _rebuild_with(self, **overrides):
        params = {
            "spots": self.spots,
            "weights": self.weights,
            "volatilities": self.volatilities,
            "correlation_matrix": self.correlation_matrix,
            "strike": self.strike,
            "time_to_expiry": self.time_to_expiry,
            "risk_free_rate": self.risk_free_rate,
            "dividend_yields": self.dividend_yields,
            "option_type": self.option_type,
            "num_paths": self.num_paths,
            "num_steps": self.num_steps,
            "random_seed": self.random_seed if self.random_seed is not None else 42,
        }
        params.update(overrides)
        return BasketOption(**params)

    def delta(self):
        """
        I define basket delta as the sensitivity of the basket price to a
        parallel 1% bump applied to every underlying simultaneously.
        """
        bump_fraction = 0.01
        spots_up = [s * (1.0 + bump_fraction) for s in self.spots]
        spots_down = [s * (1.0 - bump_fraction) for s in self.spots]

        model_up = self._rebuild_with(spots=spots_up)
        model_down = self._rebuild_with(spots=spots_down)

        weighted_bump = sum(w * (su - sd) for w, su, sd in zip(self.weights, spots_up, spots_down))
        return (model_up.price() - model_down.price()) / weighted_bump

    def gamma(self):
        bump_fraction = 0.01
        spots_up = [s * (1.0 + bump_fraction) for s in self.spots]
        spots_down = [s * (1.0 - bump_fraction) for s in self.spots]

        model_up = self._rebuild_with(spots=spots_up)
        model_down = self._rebuild_with(spots=spots_down)
        model_mid = self._rebuild_with()

        half_weighted_bump = 0.5 * sum(w * (su - sd) for w, su, sd in zip(self.weights, spots_up, spots_down))
        return (model_up.price() - 2.0 * model_mid.price() + model_down.price()) / (half_weighted_bump ** 2)

    def vega(self):
        """I bump every asset's volatility up/down by 1 vol point in parallel."""
        bump = 0.01
        vols_up = [v + bump for v in self.volatilities]
        vols_down = [max(v - bump, 1e-6) for v in self.volatilities]

        model_up = self._rebuild_with(volatilities=vols_up)
        model_down = self._rebuild_with(volatilities=vols_down)
        return (model_up.price() - model_down.price()) / (2.0 * bump)

    def theta(self):
        bump = 1.0 / 365.0
        shorter_expiry = max(self.time_to_expiry - bump, 1e-6)
        model_shorter = self._rebuild_with(time_to_expiry=shorter_expiry)
        model_base = self._rebuild_with()
        return (model_shorter.price() - model_base.price()) / bump

    def rho(self):
        bump = 0.0001
        model_up = self._rebuild_with(risk_free_rate=self.risk_free_rate + bump)
        model_down = self._rebuild_with(risk_free_rate=self.risk_free_rate - bump)
        return (model_up.price() - model_down.price()) / (2.0 * bump)

    def correlation_sensitivity(self, correlation_bump=0.10):
        """
        I bump every off-diagonal pairwise correlation up by
        `correlation_bump` (clipped to stay within [-1, 1]) and reprice the
        basket, holding every other input fixed. I return the difference
        between the bumped price and the base price, which tells me how
        exposed this basket option is to a rise in the co-movement between
        its underlyings. Basket option values generally increase with
        correlation for basket calls, since a more correlated basket has a
        wider dispersion of terminal outcomes.
        """
        base_price = self.price()

        bumped_matrix = []
        for i in range(self.num_assets):
            new_row = []
            for j in range(self.num_assets):
                if i == j:
                    new_row.append(self.correlation_matrix[i][j])
                else:
                    bumped_value = self.correlation_matrix[i][j] + correlation_bump
                    bumped_value = min(max(bumped_value, -0.999), 0.999)
                    new_row.append(bumped_value)
            bumped_matrix.append(new_row)

        model_bumped = self._rebuild_with(correlation_matrix=bumped_matrix)
        bumped_price = model_bumped.price()

        return bumped_price - base_price
