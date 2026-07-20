"""
This is my interactive Streamlit dashboard, corresponding to Phase 11 of my
project specification. I run this file with:

    streamlit run src/dashboard/app.py

I built the dashboard as a fairly plain, linear Streamlit script rather than
a class-based structure, because Streamlit's execution model re-runs the
whole script top to bottom on every user interaction, so introducing extra
abstraction layers here would not make the dashboard easier to follow. All
of the actual pricing logic lives in my src/models, src/greeks and
src/scenario_analysis modules; this file is just the UI wiring on top of
them.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import plotly.graph_objects as go

from src.models.black_scholes import BlackScholesModel
from src.models.black76 import Black76Model
from src.models.binomial_tree import CRRBinomialTree, LeisenReimerTree
from src.models.monte_carlo import AsianOption, LookbackOption, BasketOption
from src.models.finite_difference import FiniteDifferencePDE
from src.greeks.greeks_engine import GreeksEngine
from src.scenario_analysis.scenario_engine import ScenarioEngine


st.set_page_config(page_title="Derivatives Pricing Engine", layout="wide")
st.title("Derivative Pricing Engine & Greeks Calculation")
st.caption(
    "In this project, I built this dashboard as the front end for my pricing engine, "
    "Greeks engine, and scenario analysis engine."
)

PRODUCT_TO_MODELS = {
    "European Call/Put": ["Black-Scholes-Merton"],
    "American Call/Put": ["CRR Binomial Tree", "Leisen-Reimer Tree"],
    "Options on Futures": ["Black-76"],
    "Asian Option": ["Monte Carlo"],
    "Lookback Option": ["Monte Carlo"],
    "Basket Option": ["Monte Carlo"],
    "Barrier Option": ["Finite Difference PDE"],
}

with st.sidebar:
    st.header("Option Type Selector")
    product_type = st.selectbox("Product", list(PRODUCT_TO_MODELS.keys()))

    st.header("Model Selector")
    model_name = st.selectbox("Model", PRODUCT_TO_MODELS[product_type])

    st.header("Market Data Input Panel")
    option_type = st.selectbox("Option Type", ["call", "put"])
    spot = st.number_input("Underlying / Futures Price", value=100.0, min_value=0.01)
    strike = st.number_input("Strike Price", value=100.0, min_value=0.01)
    time_to_expiry = st.number_input("Time to Expiry (Years)", value=1.0, min_value=0.001)
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.05, format="%.4f")
    dividend_yield = st.number_input("Dividend Yield", value=0.02, format="%.4f")
    volatility = st.number_input("Volatility", value=0.20, min_value=0.001, format="%.4f")

    extra_params = {}
    if product_type == "American Call/Put":
        extra_params["steps"] = st.slider("Tree Steps", min_value=10, max_value=500, value=200)
    if product_type in ("Asian Option", "Lookback Option"):
        extra_params["num_paths"] = st.slider("Simulated Paths", min_value=1000, max_value=50000, value=10000, step=1000)
        extra_params["num_steps"] = st.slider("Time Steps per Path", min_value=10, max_value=252, value=50)
    if product_type == "Barrier Option":
        extra_params["barrier_level"] = st.number_input("Barrier Level", value=110.0)
        extra_params["barrier_type"] = st.selectbox(
            "Barrier Type", ["down-and-out", "up-and-out", "down-and-in", "up-and-in"]
        )


def build_model():
    if product_type == "European Call/Put":
        return BlackScholesModel(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type)

    if product_type == "Options on Futures":
        return Black76Model(spot, strike, time_to_expiry, risk_free_rate, volatility, 0.0, option_type)

    if product_type == "American Call/Put":
        if model_name == "CRR Binomial Tree":
            return CRRBinomialTree(
                spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type,
                steps=extra_params["steps"], american=True,
            )
        return LeisenReimerTree(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type,
            steps=extra_params["steps"], american=True,
        )

    if product_type == "Asian Option":
        return AsianOption(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type,
            num_paths=extra_params["num_paths"], num_steps=extra_params["num_steps"], random_seed=42,
        )

    if product_type == "Lookback Option":
        return LookbackOption(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type,
            num_paths=extra_params["num_paths"], num_steps=extra_params["num_steps"], random_seed=42,
        )

    if product_type == "Basket Option":
        return BasketOption(
            spots=[spot, spot * 0.95],
            weights=[0.5, 0.5],
            volatilities=[volatility, volatility * 1.1],
            correlation_matrix=[[1.0, 0.5], [0.5, 1.0]],
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            dividend_yields=[dividend_yield, dividend_yield],
            option_type=option_type,
            num_paths=5000,
            num_steps=50,
            random_seed=42,
        )

    if product_type == "Barrier Option":
        return FiniteDifferencePDE(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type,
            barrier_level=extra_params["barrier_level"], barrier_type=extra_params["barrier_type"],
        )

    raise ValueError(f"I do not have a model builder for product_type '{product_type}'.")


model = build_model()

col_price, col_greeks = st.columns(2)

with col_price:
    st.subheader("Pricing Output")
    st.metric("Option Price", f"{model.price():.4f}")

with col_greeks:
    st.subheader("Greeks Output")
    greeks_engine = GreeksEngine()
    greeks_table = greeks_engine.build_greeks_table({product_type: model})
    st.dataframe(greeks_table)

st.subheader("Option Payoff Diagram")
spot_range = [spot * (0.5 + 0.02 * i) for i in range(51)]
payoffs = []
for spot_at_expiry in spot_range:
    if option_type == "call":
        payoffs.append(max(spot_at_expiry - strike, 0.0) - model.price())
    else:
        payoffs.append(max(strike - spot_at_expiry, 0.0) - model.price())

payoff_figure = go.Figure()
payoff_figure.add_trace(go.Scatter(x=spot_range, y=payoffs, mode="lines", name="Net Payoff"))
payoff_figure.add_hline(y=0, line_dash="dash")
payoff_figure.update_layout(xaxis_title="Underlying Price at Expiry", yaxis_title="Net Payoff")
st.plotly_chart(payoff_figure, use_container_width=True)

st.subheader("Scenario Analysis")
scenario_engine = ScenarioEngine(model)
scenario_report = scenario_engine.full_report()

scenario_tab_names = list(scenario_report.keys())
scenario_tabs = st.tabs(scenario_tab_names)
for tab, scenario_name in zip(scenario_tabs, scenario_tab_names):
    with tab:
        st.dataframe(scenario_report[scenario_name])

if product_type == "American Call/Put":
    st.subheader("CRR vs LR Comparison")
    steps_to_compare = [10, 20, 40, 80, 160, 320]
    crr_prices = []
    lr_prices = []
    for steps in steps_to_compare:
        crr_model = CRRBinomialTree(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type, steps=steps)
        lr_model = LeisenReimerTree(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type, steps=steps)
        crr_prices.append(crr_model.price())
        lr_prices.append(lr_model.price())

    comparison_figure = go.Figure()
    comparison_figure.add_trace(go.Scatter(x=steps_to_compare, y=crr_prices, mode="lines+markers", name="CRR"))
    comparison_figure.add_trace(go.Scatter(x=steps_to_compare, y=lr_prices, mode="lines+markers", name="Leisen-Reimer"))
    comparison_figure.update_layout(xaxis_title="Steps", yaxis_title="Option Price")
    st.plotly_chart(comparison_figure, use_container_width=True)

if product_type in ("Asian Option", "Lookback Option"):
    st.subheader("Monte Carlo Convergence Chart")
    path_counts_to_compare = [500, 1000, 2000, 4000, 8000]
    mc_prices = []
    for path_count in path_counts_to_compare:
        mc_model = type(model)(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type,
            num_paths=path_count, num_steps=extra_params["num_steps"], random_seed=42,
        )
        mc_prices.append(mc_model.price())

    convergence_figure = go.Figure()
    convergence_figure.add_trace(go.Scatter(x=path_counts_to_compare, y=mc_prices, mode="lines+markers", name="MC Price"))
    convergence_figure.update_layout(xaxis_title="Number of Paths", yaxis_title="Option Price")
    st.plotly_chart(convergence_figure, use_container_width=True)

st.subheader("Volatility Smile Visualization")
st.caption("I plot a synthetic smile here from manually entered strikes/implied vols since no live data feed is connected in this environment.")
smile_strikes = [strike * (0.8 + 0.05 * i) for i in range(9)]
smile_vols = [volatility * (1.15 - 0.03 * i) for i in range(9)]
smile_figure = go.Figure()
smile_figure.add_trace(go.Scatter(x=smile_strikes, y=[v * 100 for v in smile_vols], mode="lines+markers"))
smile_figure.update_layout(xaxis_title="Strike", yaxis_title="Implied Volatility (%)")
st.plotly_chart(smile_figure, use_container_width=True)

st.subheader("Risk Metrics Panel")
risk_columns = st.columns(5)
greek_labels = ["delta", "gamma", "vega", "theta", "rho"]
for column, greek_label in zip(risk_columns, greek_labels):
    column.metric(greek_label.capitalize(), f"{getattr(model, greek_label)():.4f}")
