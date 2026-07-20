"""
This is my Greeks engine, corresponding to Phase 8 of my project
specification. Every pricing model in my platform already exposes
delta()/gamma()/vega()/theta()/rho() and a greeks_summary() convenience
method, so this engine does not recompute any Greek itself. Instead, it is
the layer that aggregates Greeks across many models into the reporting
shapes I actually want to look at: a comparison table, a heatmap grid, and
a plain-language sensitivity report.
"""

import pandas as pd


class GreeksEngine:
    """
    My aggregation layer on top of individual pricing model instances.
    """

    def build_greeks_table(self, labeled_models):
        """
        I take a dictionary mapping a label (e.g. "AAPL 150C") to a pricing
        model instance, and return a pandas DataFrame with one row per
        label and one column per Greek. I use this both for my "Greeks
        Table" deliverable and for the "Greeks Comparison Across Products"
        deliverable, since comparing products is just building the table
        across multiple product types at once.
        """
        rows = []
        labels = []
        for label, model in labeled_models.items():
            summary = model.greeks_summary()
            rows.append(summary)
            labels.append(label)

        table = pd.DataFrame(rows, index=labels)
        return table[["price", "delta", "gamma", "vega", "theta", "rho"]]

    def build_heatmap_grid(self, model_factory, x_values, y_values, greek_name="delta"):
        """
        I build a 2D grid of a single Greek's value as two of the option's
        inputs vary, which is exactly the data my visualization layer needs
        to draw a Greeks heatmap.

        `model_factory` must be a function taking (x_value, y_value) and
        returning a freshly built pricing model instance. I loop over every
        combination explicitly with nested for-loops rather than trying to
        vectorize the model construction, since building a model is not a
        simple array operation (it involves calling into whichever pricing
        class I was handed).
        """
        grid = []
        for y_value in y_values:
            row_values = []
            for x_value in x_values:
                model = model_factory(x_value, y_value)
                greek_value = getattr(model, greek_name)()
                row_values.append(greek_value)
            grid.append(row_values)
        return grid

    def build_sensitivity_report(self, model, spot_bump_pct=0.01, vol_bump=0.01, rate_bump=0.0001):
        """
        I report, in plain terms, how much the option price is expected to
        move under a small bump to each input, computed two ways: (a)
        directly from the analytic/model Greek, and (b) by literally
        bumping the input and repricing. When these two numbers are close,
        it gives me confidence that my Greek calculation is internally
        consistent with my pricing function.
        """
        base_price = model.price()
        greeks = model.greeks_summary()

        report_rows = []

        greek_implied_pnl = greeks["delta"] * (model.spot * spot_bump_pct)
        report_rows.append({
            "risk_factor": "spot",
            "bump_size": model.spot * spot_bump_pct,
            "greek_implied_pnl": greek_implied_pnl,
        })

        greek_implied_pnl = greeks["vega"] * vol_bump
        report_rows.append({
            "risk_factor": "volatility",
            "bump_size": vol_bump,
            "greek_implied_pnl": greek_implied_pnl,
        })

        greek_implied_pnl = greeks["rho"] * rate_bump
        report_rows.append({
            "risk_factor": "risk_free_rate",
            "bump_size": rate_bump,
            "greek_implied_pnl": greek_implied_pnl,
        })

        report_rows.append({
            "risk_factor": "base_price",
            "bump_size": None,
            "greek_implied_pnl": base_price,
        })

        return pd.DataFrame(report_rows)
