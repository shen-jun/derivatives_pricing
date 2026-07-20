"""
This is my visualization module, covering the chart deliverables listed
across Phase 3 through Phase 10 of my project specification.

I built every chart function around the same idea: I pass in a
`model_factory`, which is a plain Python function that takes one input
value (e.g. a spot price) and returns a freshly built pricing model
instance for that input. This lets the same `plot_price_vs_spot` function
work whether I am plotting a BlackScholesModel, a Black76Model, a
CRRBinomialTree, or anything else in my platform, without duplicating the
plotting logic per model type.

I use matplotlib with the non-interactive "Agg" backend so these functions
run cleanly inside my test suite and inside a headless server, and I return
the Figure object from every function so the caller (my dashboard, or a
script saving PNGs) decides what to do with it.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_price_vs_spot(model_factory, spot_values, title="Option Price vs Spot Price"):
    prices = []
    for spot in spot_values:
        model = model_factory(spot)
        prices.append(model.price())

    fig, ax = plt.subplots()
    ax.plot(spot_values, prices, color="tab:blue")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Option Price")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_price_vs_strike(model_factory, strike_values, title="Option Price vs Strike Price"):
    prices = []
    for strike in strike_values:
        model = model_factory(strike)
        prices.append(model.price())

    fig, ax = plt.subplots()
    ax.plot(strike_values, prices, color="tab:orange")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Option Price")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_price_vs_volatility(model_factory, volatility_values, title="Option Price vs Volatility"):
    prices = []
    for volatility in volatility_values:
        model = model_factory(volatility)
        prices.append(model.price())

    fig, ax = plt.subplots()
    ax.plot([v * 100 for v in volatility_values], prices, color="tab:green")
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Option Price")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_price_vs_time_to_expiry(model_factory, time_values, title="Option Price vs Time to Expiry"):
    prices = []
    for time_to_expiry in time_values:
        model = model_factory(time_to_expiry)
        prices.append(model.price())

    fig, ax = plt.subplots()
    ax.plot(time_values, prices, color="tab:red")
    ax.set_xlabel("Time to Expiry (Years)")
    ax.set_ylabel("Option Price")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_greek_vs_input(model_factory, input_values, greek_name, x_label, title=None):
    """
    I use this one generic function for every "Greek vs Something" chart in
    my specification: Delta vs Stock Price, Gamma vs Stock Price, Vega vs
    Volatility, Theta vs Time, Rho vs Interest Rate. The only thing that
    changes between those charts is which input I sweep and which Greek I
    read off the model, so I parametrized both instead of writing five
    near-identical functions.
    """
    greek_values = []
    for input_value in input_values:
        model = model_factory(input_value)
        greek_values.append(getattr(model, greek_name)())

    if title is None:
        title = f"{greek_name.capitalize()} vs {x_label}"

    fig, ax = plt.subplots()
    ax.plot(input_values, greek_values, color="tab:purple")
    ax.set_xlabel(x_label)
    ax.set_ylabel(greek_name.capitalize())
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_convergence(step_values, price_values, benchmark_price=None, title="Price Convergence vs Number of Steps"):
    """
    I use this for CRR convergence, LR convergence, and Monte Carlo
    convergence charts alike: all three just need step/path count on the
    x-axis and the resulting price estimate on the y-axis, optionally with
    a horizontal reference line for the true or benchmark price.
    """
    fig, ax = plt.subplots()
    ax.plot(step_values, price_values, marker="o", color="tab:blue", label="Model Price")

    if benchmark_price is not None:
        ax.axhline(benchmark_price, color="tab:red", linestyle="--", label="Benchmark Price")

    ax.set_xlabel("Number of Steps / Paths")
    ax.set_ylabel("Option Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig


def plot_pricing_error(step_values, error_values, title="Pricing Error vs Number of Steps"):
    fig, ax = plt.subplots()
    ax.plot(step_values, error_values, marker="o", color="tab:red")
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Absolute Pricing Error")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_model_comparison(step_values, price_series_by_label, title="Model Comparison"):
    """
    I use this both for "American vs European Price Comparison" and for
    "CRR vs LR Comparison": in both cases I just have two or more labeled
    series of prices plotted against the same x-axis (steps, or spot).
    """
    fig, ax = plt.subplots()
    for label, price_values in price_series_by_label.items():
        ax.plot(step_values, price_values, marker="o", label=label)

    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Option Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig


def plot_early_exercise_boundary(boundary_points, title="Early Exercise Boundary"):
    times = []
    critical_prices = []
    for time_value, critical_price in boundary_points:
        if critical_price is not None:
            times.append(time_value)
            critical_prices.append(critical_price)

    fig, ax = plt.subplots()
    ax.plot(times, critical_prices, color="tab:brown")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Critical Asset Price")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_simulated_paths(paths, title="Simulated Price Paths"):
    """
    I plot every simulated Monte Carlo path (a list of price lists) on the
    same axes with light, semi-transparent lines so the overall spread of
    outcomes is visible at a glance.
    """
    fig, ax = plt.subplots()
    for path in paths:
        time_axis = list(range(len(path)))
        ax.plot(time_axis, path, color="tab:blue", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Simulated Price")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_confidence_interval_bands(path_count_values, price_values, lower_bounds, upper_bounds, title="Monte Carlo Confidence Interval Bands"):
    fig, ax = plt.subplots()
    ax.plot(path_count_values, price_values, color="tab:blue", label="MC Price Estimate")
    ax.fill_between(path_count_values, lower_bounds, upper_bounds, color="tab:blue", alpha=0.2, label="95% Confidence Band")
    ax.set_xlabel("Number of Simulated Paths")
    ax.set_ylabel("Option Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig


def plot_payoff_distribution(payoffs, title="Payoff Distribution", num_bins=40):
    fig, ax = plt.subplots()
    ax.hist(payoffs, bins=num_bins, color="tab:green", edgecolor="black")
    ax.set_xlabel("Simulated Payoff")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_correlation_sensitivity(correlation_values, price_values, title="Basket Correlation Sensitivity"):
    fig, ax = plt.subplots()
    ax.plot(correlation_values, price_values, marker="o", color="tab:orange")
    ax.set_xlabel("Pairwise Correlation")
    ax.set_ylabel("Basket Option Price")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_price_heatmap(x_values, y_values, price_grid, x_label, y_label, title="Price Heatmap"):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(
        price_grid,
        aspect="auto",
        origin="lower",
        extent=[min(x_values), max(x_values), min(y_values), max(y_values)],
        cmap="viridis",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(heatmap, ax=ax, label="Option Price")
    return fig


def plot_barrier_sensitivity(barrier_levels, price_values, title="Option Value vs Barrier Level"):
    fig, ax = plt.subplots()
    ax.plot(barrier_levels, price_values, color="tab:red")
    ax.set_xlabel("Barrier Level")
    ax.set_ylabel("Option Price")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_pnl_surface(spot_values, vol_values, pnl_grid, title="P&L Surface"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    spot_mesh = []
    vol_mesh = []
    for vol in vol_values:
        spot_row = []
        vol_row = []
        for spot in spot_values:
            spot_row.append(spot)
            vol_row.append(vol)
        spot_mesh.append(spot_row)
        vol_mesh.append(vol_row)

    ax.plot_surface(spot_mesh, vol_mesh, pnl_grid, cmap="coolwarm")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    ax.set_zlabel("P&L")
    ax.set_title(title)
    return fig


def plot_greeks_heatmap(x_values, y_values, greek_grid, x_label, y_label, greek_name, title=None):
    if title is None:
        title = f"{greek_name.capitalize()} Heatmap"
    return plot_price_heatmap(x_values, y_values, greek_grid, x_label, y_label, title)


def plot_payoff_diagram(spot_values, strike, option_type="call", premium=0.0, title="Option Payoff Diagram"):
    """
    I plot the classic hockey-stick payoff diagram at expiry, net of the
    premium paid, which is the payoff diagram my dashboard shows for
    whichever product the user selects.
    """
    payoffs = []
    for spot in spot_values:
        if option_type == "call":
            intrinsic_value = max(spot - strike, 0.0)
        else:
            intrinsic_value = max(strike - spot, 0.0)
        payoffs.append(intrinsic_value - premium)

    fig, ax = plt.subplots()
    ax.plot(spot_values, payoffs, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Underlying Price at Expiry")
    ax.set_ylabel("Net Payoff")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_delta_gamma_approximation(model, spot_shock_range, title="Delta-Gamma Approximation vs Actual Repricing"):
    """
    I compare the actual repriced P&L against the second-order Taylor
    (delta-gamma) approximation:

        Approx P&L = Delta * dS + 0.5 * Gamma * dS^2

    across a range of spot shocks, holding every other input fixed. This is
    the standard way I check how good a linear/quadratic risk approximation
    is compared to a full reprice, which matters a lot for how much I trust
    a risk desk's delta-gamma P&L explain.
    """
    import copy

    base_price = model.price()
    delta = model.delta()
    gamma = model.gamma()

    actual_pnl = []
    approx_pnl = []
    for shock in spot_shock_range:
        dS = model.spot * shock
        cloned_model = copy.deepcopy(model)
        cloned_model.spot = model.spot + dS
        actual_pnl.append(cloned_model.price() - base_price)
        approx_pnl.append(delta * dS + 0.5 * gamma * dS * dS)

    fig, ax = plt.subplots()
    ax.plot([s * 100 for s in spot_shock_range], actual_pnl, label="Actual Repricing", color="tab:blue")
    ax.plot([s * 100 for s in spot_shock_range], approx_pnl, label="Delta-Gamma Approximation", linestyle="--", color="tab:red")
    ax.set_xlabel("Spot Shock (%)")
    ax.set_ylabel("P&L")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig


def plot_scenario_shocks(scenario_rows, x_key, y_key, title="Scenario Shock Chart"):
    """
    I use this to plot any of my ScenarioEngine report outputs (price
    shocks, vol shocks, rate shocks, time decay) as a simple bar chart of
    P&L by shock size.
    """
    x_values = [row[x_key] for row in scenario_rows]
    y_values = [row[y_key] for row in scenario_rows]

    fig, ax = plt.subplots()
    bar_colors = ["tab:green" if value >= 0 else "tab:red" for value in y_values]
    ax.bar([str(x) for x in x_values], y_values, color=bar_colors)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, axis="y")
    return fig


def plot_volatility_smile(strike_values, implied_vol_values, title="Volatility Smile"):
    fig, ax = plt.subplots()
    ax.plot(strike_values, [v * 100 for v in implied_vol_values], marker="o", color="tab:purple")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(title)
    ax.grid(True)
    return fig
