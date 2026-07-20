"""
This is my scenario analysis engine, corresponding to Phase 9 of my project
specification. I deliberately made this engine work generically against any
of my pricing model classes (Black-Scholes-Merton, Black-76, CRR/LR trees,
Monte Carlo, Finite Difference PDE, and Basket) by cloning the model with
`copy.deepcopy` and then mutating its stored attributes directly, instead of
re-instantiating each model class with its own constructor signature. Since
every model recomputes its price from its stored attributes whenever
price() is called, mutating a cloned copy and re-pricing it behaves exactly
like re-running the whole model with a shocked input.
"""

import copy


class ScenarioEngine:
    """
    My scenario/shock engine wrapping a single pricing model instance.
    """

    def __init__(self, model):
        self.model = model

    def _clone(self):
        return copy.deepcopy(self.model)

    def _is_basket(self, model):
        return hasattr(model, "spots") and hasattr(model, "weights")

    def apply_price_shock(self, shock_pct):
        """I shock the underlying price (or every basket constituent in parallel) by shock_pct."""
        cloned = self._clone()
        if self._is_basket(cloned):
            cloned.spots = [s * (1.0 + shock_pct) for s in cloned.spots]
            cloned.spot = sum(w * s for w, s in zip(cloned.weights, cloned.spots))
        else:
            cloned.spot = cloned.spot * (1.0 + shock_pct)
        return cloned.price()

    def underlying_price_shock_report(self):
        """
        Phase 9 requires shocks of +1%, -1%, +5%, -5%, +10%, -10% applied to
        the underlying price.
        """
        shocks = [0.01, -0.01, 0.05, -0.05, 0.10, -0.10]
        base_price = self._clone().price()

        rows = []
        for shock in shocks:
            shocked_price = self.apply_price_shock(shock)
            rows.append({
                "shock_pct": shock * 100.0,
                "shocked_price": shocked_price,
                "pnl": shocked_price - base_price,
            })
        return rows

    def apply_vol_shock(self, vol_points):
        """I shock volatility (or every basket volatility in parallel) by vol_points (e.g. 0.05 = 5 vol points)."""
        cloned = self._clone()
        if self._is_basket(cloned):
            cloned.volatilities = [max(v + vol_points, 1e-6) for v in cloned.volatilities]
        else:
            cloned.volatility = max(cloned.volatility + vol_points, 1e-6)
        return cloned.price()

    def volatility_shock_report(self):
        """
        Phase 9 requires shocks of +5, -5, +10, -10 vol points.
        """
        shocks = [0.05, -0.05, 0.10, -0.10]
        base_price = self._clone().price()

        rows = []
        for shock in shocks:
            shocked_price = self.apply_vol_shock(shock)
            rows.append({
                "vol_shock_points": shock * 100.0,
                "shocked_price": shocked_price,
                "pnl": shocked_price - base_price,
            })
        return rows

    def apply_rate_shock(self, basis_points):
        cloned = self._clone()
        cloned.risk_free_rate = cloned.risk_free_rate + basis_points / 10000.0
        return cloned.price()

    def rate_shock_report(self):
        """
        Phase 9 requires shocks of +50bps, -50bps, +100bps, -100bps.
        """
        shocks_bps = [50, -50, 100, -100]
        base_price = self._clone().price()

        rows = []
        for basis_points in shocks_bps:
            shocked_price = self.apply_rate_shock(basis_points)
            rows.append({
                "rate_shock_bps": basis_points,
                "shocked_price": shocked_price,
                "pnl": shocked_price - base_price,
            })
        return rows

    def apply_time_decay(self, calendar_days):
        cloned = self._clone()
        cloned.time_to_expiry = max(cloned.time_to_expiry - calendar_days / 365.0, 1e-6)
        return cloned.price()

    def time_decay_report(self):
        """
        Phase 9 requires time decay analysis over 1, 7 and 30 calendar days.
        """
        day_offsets = [1, 7, 30]
        base_price = self._clone().price()

        rows = []
        for days in day_offsets:
            decayed_price = self.apply_time_decay(days)
            rows.append({
                "days_forward": days,
                "shocked_price": decayed_price,
                "pnl": decayed_price - base_price,
            })
        return rows

    def correlation_shock_report(self, shocks=(0.05, -0.05, 0.10, -0.10)):
        """
        I only support this for a BasketOption, since correlation only
        exists as an input on basket products in my platform.
        """
        if not hasattr(self.model, "correlation_matrix"):
            raise ValueError("I can only run a correlation shock report on a BasketOption model.")

        rows = []
        for shock in shocks:
            cloned = self._clone()
            pnl_change = cloned.correlation_sensitivity(correlation_bump=shock)
            rows.append({"correlation_shock": shock, "pnl": pnl_change})
        return rows

    def barrier_level_shock_report(self, shocks=(0.05, -0.05, 0.10, -0.10)):
        """
        I only support this for a FiniteDifferencePDE barrier model.
        """
        if not hasattr(self.model, "barrier_level"):
            raise ValueError("I can only run a barrier level shock report on a FiniteDifferencePDE model.")

        rows = []
        for shock in shocks:
            cloned = self._clone()
            pnl_change = cloned.barrier_sensitivity(barrier_bump=shock)
            rows.append({"barrier_shock": shock, "pnl": pnl_change})
        return rows

    def full_report(self):
        """
        I bundle every shock category that applies to the wrapped model
        into one dictionary of lists, which my dashboard renders as a set
        of tables/charts side by side.
        """
        report = {
            "underlying_price_shocks": self.underlying_price_shock_report(),
            "volatility_shocks": self.volatility_shock_report(),
            "rate_shocks": self.rate_shock_report(),
            "time_decay": self.time_decay_report(),
        }

        if hasattr(self.model, "correlation_matrix"):
            report["correlation_shocks"] = self.correlation_shock_report()

        if hasattr(self.model, "barrier_level"):
            report["barrier_shocks"] = self.barrier_level_shock_report()

        return report
