"""
My smoke tests for the visualization layer. I am not trying to verify pixel
output here, just that every chart function in charts.py runs without
raising and returns a matplotlib Figure, since a broken chart function
would otherwise only be caught by a human looking at the dashboard.
"""

import matplotlib.figure

from src.models.black_scholes import BlackScholesModel
from src.models.binomial_tree import CRRBinomialTree
from src.visualization import charts


def test_plot_price_vs_spot_returns_a_figure():
    def model_factory(spot):
        return BlackScholesModel(spot, 100, 1.0, 0.05, 0.20, option_type="call")

    figure = charts.plot_price_vs_spot(model_factory, [80, 90, 100, 110, 120])
    assert isinstance(figure, matplotlib.figure.Figure)


def test_plot_greek_vs_input_returns_a_figure():
    def model_factory(spot):
        return BlackScholesModel(spot, 100, 1.0, 0.05, 0.20, option_type="call")

    figure = charts.plot_greek_vs_input(model_factory, [80, 90, 100, 110, 120], "delta", "Spot Price")
    assert isinstance(figure, matplotlib.figure.Figure)


def test_plot_convergence_returns_a_figure():
    figure = charts.plot_convergence([10, 50, 100], [4.5, 4.7, 4.75], benchmark_price=4.76)
    assert isinstance(figure, matplotlib.figure.Figure)


def test_plot_early_exercise_boundary_returns_a_figure():
    tree = CRRBinomialTree(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.03, option_type="put", steps=50, american=True)
    boundary_points = tree.early_exercise_boundary()

    figure = charts.plot_early_exercise_boundary(boundary_points)
    assert isinstance(figure, matplotlib.figure.Figure)


def test_plot_payoff_diagram_returns_a_figure():
    figure = charts.plot_payoff_diagram([80, 90, 100, 110, 120], strike=100, option_type="call", premium=3.5)
    assert isinstance(figure, matplotlib.figure.Figure)


def test_plot_delta_gamma_approximation_returns_a_figure():
    model = BlackScholesModel(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.20, option_type="call")
    figure = charts.plot_delta_gamma_approximation(model, [-0.10, -0.05, 0.0, 0.05, 0.10])
    assert isinstance(figure, matplotlib.figure.Figure)
