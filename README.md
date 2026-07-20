# Derivative Pricing Engine & Greeks Calculation

In this project, I built a Python derivatives analytics platform that prices seven option types across five different pricing techniques, computes their Greeks, runs scenario/shock analysis on them, visualizes the results, and exposes everything through an interactive Streamlit dashboard. This README documents the folder structure I designed, how to run everything, and what every single file, class, and method in my codebase actually does.

I wrote the mathematical derivation behind every formula in `math_and_logic.tex`. This README focuses on the code itself; I only restate a formula here when it helps explain a design decision.

## 1. Folder Structure I Designed

```
derivatives_pricing/
├── requirements.txt
├── README.md
├── math_and_logic.tex
├── math_and_logic.pdf
├── conftest.py
├── src/
│   ├── models/
│   │   ├── base_model.py
│   │   ├── math_utils.py
│   │   ├── black_scholes.py
│   │   ├── black76.py
│   │   ├── binomial_tree.py
│   │   ├── monte_carlo.py
│   │   └── finite_difference.py
│   ├── calibration/
│   │   └── implied_volatility.py
│   ├── market_data/
│   │   └── data_fetcher.py
│   ├── greeks/
│   │   └── greeks_engine.py
│   ├── scenario_analysis/
│   │   └── scenario_engine.py
│   ├── visualization/
│   │   └── charts.py
│   └── dashboard/
│       └── app.py
└── tests/
    ├── test_black_scholes.py
    ├── test_black76.py
    ├── test_binomial_tree.py
    ├── test_monte_carlo.py
    ├── test_finite_difference.py
    ├── test_calibration.py
    ├── test_market_data.py
    ├── test_greeks_engine.py
    ├── test_scenario_analysis.py
    ├── test_visualization.py
    └── test_model_validation.py
```

I organized `src/` around the same buckets the project specification uses (`models/`, `greeks/`, `calibration/`, `market_data/`, `scenario_analysis/`, `visualization/`, `dashboard/`), so that each folder maps directly onto one phase of the specification. I kept `tests/` as one flat folder rather than mirroring the `src/` tree one-for-one, since most of my test files test a single module and a nested test tree would just add navigation overhead for no benefit.

## 2. Execution Instructions

I built and tested this project against Python 3.10.

### 2.1 Install dependencies

```bash
cd derivatives_pricing
pip install -r requirements.txt
```

### 2.2 Run the unit test suite

```bash
pytest tests/ -v
```

I added `conftest.py` at the project root specifically so pytest adds the project root to `sys.path` before it collects my tests, since every test module imports from `src` using an absolute import like `from src.models.black_scholes import BlackScholesModel`. As long as I run `pytest` from inside `derivatives_pricing/` (or point it at this folder), this works with no extra configuration.

### 2.3 Launch the interactive dashboard

```bash
streamlit run src/dashboard/app.py
```

## 3. Deep Dive: What Every File Does

### 3.1 `src/models/base_model.py` — `OptionModel`

This is the abstract base class every single pricing model in my platform inherits from. I designed it around one idea: whatever pricing technique sits underneath a contract, the rest of my platform (the Greeks engine, the scenario engine, the dashboard) should be able to call `price()`, `delta()`, `gamma()`, `vega()`, `theta()`, and `rho()` on it without knowing or caring which model it is talking to. I store the shared contract/market inputs (spot, strike, time to expiry, risk-free rate, volatility, dividend yield, option type) on the instance in `__init__`, with validation that rejects non-positive spot/strike and an unrecognized option type immediately. I also give every model a `greeks_summary()` convenience method (returns a dict of all six values in one call) and an `intrinsic_value()` helper that several subclasses reuse for early-exercise checks and payoff calculations.

### 3.2 `src/models/math_utils.py`

I put my small shared math helpers here so I don't repeat the same normal-distribution formulas across `black_scholes.py`, `black76.py`, and `implied_volatility.py`. `normal_cdf()` and `normal_pdf()` compute $N(x)$ and $n(x)$ by hand (using `math.erf` rather than importing `scipy.stats.norm`), and `d1_d2()` computes the shared $d_1$/$d_2$ pair that every Black-Scholes-style formula in this project is built from.

### 3.3 `src/models/black_scholes.py` — `BlackScholesModel`

My closed-form pricer for European calls and puts with continuous dividend yield support. Every method (`price`, `delta`, `gamma`, `vega`, `theta`, `rho`) is a direct, line-by-line implementation of the corresponding formula in Section 2 of `math_and_logic.tex`. I handle the `time_to_expiry == 0` edge case explicitly in every method (an expired option just returns its intrinsic value/limiting Greek, since $d_1$/$d_2$ are undefined when $T=0$).

### 3.4 `src/models/black76.py` — `Black76Model`

My closed-form pricer for European options on futures/forwards. I reuse `self.spot` to hold the futures price $F$ and I deliberately ignore `self.dividend_yield` here (documented clearly in the class docstring), since a futures price already embeds its own cost of carry. The one thing worth calling out is `rho()`: because $r$ only ever enters the Black-76 formula through the discount factor applied to the whole expectation, I implement it as the one-line identity `rho = -T * price()` rather than expanding it out algebraically.

### 3.5 `src/models/binomial_tree.py` — `BinomialTreeModel`, `CRRBinomialTree`, `LeisenReimerTree`

`BinomialTreeModel` is my shared base class for both tree constructions. It owns:

- `_build_full_grid()`: builds the entire backward-induction grid as a list of lists using explicit nested `for` loops (not a vectorized numpy array), applying the early-exercise check at every node when `american=True`. I keep the *full* grid in memory (not just the root price) because I reuse the early nodes to extract Greeks.
- `price()`, `delta()`, `gamma()`, `theta()`: read straight off the grid nodes, following the standard technique described in Hull's textbook.
- `vega()`, `rho()`: bump the corresponding input, rebuild an entirely new tree via `_rebuild_with()`, and take a central finite difference, since trees don't hand me a clean closed-form vega/rho.
- `early_exercise_boundary()`: scans every time step for the critical asset price where immediate exercise first becomes optimal, which I use for my "Early Exercise Boundary" chart. This only makes sense for `american=True`, so I raise `ValueError` otherwise.

`CRRBinomialTree` and `LeisenReimerTree` each override one method, `_tree_parameters()`, returning their own $(\Delta t, u, d, p)$. `LeisenReimerTree` additionally forces the step count to be odd in `__init__`, and implements the Peizer-Pratt inversion formula as a static method.

### 3.6 `src/models/monte_carlo.py` — `MonteCarloOptionModel`, `AsianOption`, `LookbackOption`, `BasketOption`

`MonteCarloOptionModel` is my base class for single-underlying, path-dependent options. `_simulate_path_from_normals()` rolls the GBM recursion forward one time step at a time with an explicit `for` loop over a list of normal draws — I wrote it this way specifically so the exact sequence of multiplications is visible, rather than hidden inside a vectorized cumulative product. `price()` implements antithetic variates (for every set of normal draws I also build the path from the negated draws) and stores both the discounted price and its standard error, so `confidence_interval()` can report a 95% band on demand. `delta`, `gamma`, `vega`, `theta`, `rho` are all bump-and-reprice finite differences that reuse the same `random_seed` between the bumped and base models (common random numbers), which is what keeps a Monte Carlo Greek from being swamped by simulation noise.

`AsianOption` and `LookbackOption` each only need to override `_payoff_from_path()`. `LookbackOption` additionally supports a `floating_strike` flag to switch between the fixed-strike and floating-strike payoff conventions.

`BasketOption` does not inherit the single-asset constructor signature, since a basket genuinely needs a list of spots, weights, volatilities, dividend yields, and a full correlation matrix. It implements its own `_cholesky_decomposition()` (explicit nested loops, no numpy `linalg`) to correlate independent normal draws across assets at every time step, and its own bump-and-reprice Greeks (bumping every asset in parallel for delta/gamma/vega). It also exposes `correlation_sensitivity()`, which bumps every off-diagonal correlation entry and reports the resulting price change — this is the method my scenario engine and dashboard call for the "Basket Correlation Sensitivity" deliverable.

### 3.7 `src/models/finite_difference.py` — `FiniteDifferencePDE`

My explicit finite difference solver for single-barrier options. `_price_knock_out_grid()` builds the full $(S, t)$ grid: terminal payoffs at maturity, then backward induction using the explicit update coefficients $a_i, b_i, c_i$ derived in `math_and_logic.tex`, zeroing out any node that has crossed the barrier at every time step. `_stable_num_time_steps()` automatically increases the number of time steps if the caller's choice would violate my stability bound, so the scheme can't silently return garbage from an unstable configuration. Knock-in contracts are priced via the in/out parity identity (`vanilla - knock_out`) rather than a separate grid. `delta()`, `gamma()`, `theta()` read off neighboring grid nodes around the (linearly interpolated) spot index, falling back to bump-and-reprice near the edge of the grid; `vega()` and `rho()` are always bump-and-reprice. `barrier_sensitivity()` bumps the barrier level and reports the resulting price change, which feeds my "Barrier Sensitivity" and "Option Value vs Barrier Level" charts.

### 3.8 `src/calibration/implied_volatility.py` — `ImpliedVolatilitySolver`

My implied volatility calibration engine. `solve()` first checks the market price against the Black-Scholes no-arbitrage bounds (computed in `_no_arbitrage_bounds()`) and raises immediately if the price is impossible to reproduce at any volatility. It then tries `_newton_raphson()`, which iterates $\sigma_{k+1} = \sigma_k - (C(\sigma_k) - C_{market})/\mathcal{V}(\sigma_k)$ using the model's own vega, and returns `None` if it ever steps outside my volatility bounds or vega gets too small to safely divide by. If Newton-Raphson gives up, `solve()` falls back to `_bisection()`, a plain bracketing search over `[min_volatility, max_volatility]` that is slower but guaranteed to converge as long as the market price is bracketed.

### 3.9 `src/market_data/data_fetcher.py` — `MarketDataPoint`, `YahooFinanceFetcher`

`MarketDataPoint` is a plain dataclass holding every market input listed in Phase 1 of the specification (underlying price, strike, expiry, rate, dividend yield, implied vol, market price, bid/ask, volume, open interest), with `bid_ask_spread` computed as a property rather than stored redundantly. `YahooFinanceFetcher` wraps the `yfinance` package for `fetch_spot_price()` and `fetch_option_chain()`, translating every row of the option chain into a `MarketDataPoint`. I import `yfinance` inside the constructor (not at module import time) specifically so the rest of my codebase — and my whole test suite — can be imported and run in an environment with no network access or without `yfinance` installed. `build_manual_market_data_point()` is the helper I actually use in my tests, examples, and dashboard defaults, since it does not depend on any network call.

### 3.10 `src/greeks/greeks_engine.py` — `GreeksEngine`

This is my Greeks aggregation layer. It does not compute any Greek itself — every pricing model already does that — it just reshapes Greeks from one or many model instances into the views I actually want to look at. `build_greeks_table()` turns a `{label: model}` dictionary into a pandas DataFrame (one row per label), which serves both the "Greeks Table" and "Greeks Comparison Across Products" deliverables. `build_heatmap_grid()` takes a `model_factory(x, y)` function and two lists of input values, and returns a 2D grid of a chosen Greek's value, which feeds my Greeks heatmap chart. `build_sensitivity_report()` reports the P&L implied by each Greek under a small bump to spot/vol/rate, alongside the base price, as a sanity-check table.

### 3.11 `src/scenario_analysis/scenario_engine.py` — `ScenarioEngine`

My shock/scenario engine, built to work generically against any model in my platform. Rather than reconstructing a new model instance with a model-specific constructor, I clone the model with `copy.deepcopy()` and mutate the clone's stored attributes directly (`spot`, `volatility`, `risk_free_rate`, `time_to_expiry`, or the basket-specific `spots`/`volatilities` lists), then call `price()` again — since every model recomputes its price from its own attributes, this behaves identically to a full re-run. I implement one method per shock category from Phase 9 of the specification (`underlying_price_shock_report`, `volatility_shock_report`, `rate_shock_report`, `time_decay_report`), plus `correlation_shock_report()` and `barrier_level_shock_report()`, which only run against a `BasketOption` or `FiniteDifferencePDE` respectively (checked via `hasattr`). `full_report()` bundles whichever of these apply to the wrapped model into one dictionary.

### 3.12 `src/visualization/charts.py`

Every function here returns a matplotlib `Figure` (using the non-interactive `Agg` backend, so these run cleanly in my test suite and on a headless server). I built most of these functions to be generic over a `model_factory` callable — a plain function that takes one input value and returns a freshly built model — so the same `plot_price_vs_spot()` function works whether I'm plotting a `BlackScholesModel`, a `Black76Model`, or a `CRRBinomialTree`. `plot_greek_vs_input()` is the one function behind five different chart requirements in the specification (Delta vs Spot, Gamma vs Spot, Vega vs Vol, Theta vs Time, Rho vs Rate), since the only thing that changes between them is which input I sweep and which Greek I read off. I also have dedicated functions for convergence charts, the early exercise boundary, simulated Monte Carlo paths, confidence interval bands, payoff distributions, correlation sensitivity, price/Greeks heatmaps, barrier sensitivity, P&L surfaces, payoff diagrams, the delta-gamma Taylor approximation check, generic scenario shock bar charts, and a volatility smile plot.

### 3.13 `src/dashboard/app.py`

My Streamlit dashboard, run with `streamlit run src/dashboard/app.py`. I wrote this as a fairly plain linear script rather than introducing extra class structure, since Streamlit re-runs the whole script top to bottom on every interaction, so the usual OOP encapsulation benefits don't really apply here. The sidebar implements the option type selector, model selector, and full market data input panel from Phase 11 of the specification. `build_model()` maps the selected product/model combination to the right class from `src/models/`. The main panel then renders: pricing output, Greeks output (via `GreeksEngine`), a Plotly payoff diagram, a tabbed scenario analysis view (via `ScenarioEngine`), a CRR vs LR comparison chart (for American options), a Monte Carlo convergence chart (for Asian/Lookback options), a synthetic volatility smile plot, and a risk metrics panel showing all five Greeks as Streamlit metric widgets.

### 3.14 `tests/`

Every pricing model has its own test file checking known reference values, put-call parity where applicable, and that my analytic/tree/grid Greeks agree with an independent bump-and-reprice finite difference. `test_calibration.py` checks that my implied volatility solver recovers a volatility I deliberately priced an option at. `test_market_data.py` covers the manual `MarketDataPoint` builder (I don't test the live Yahoo Finance path, since my test suite needs to run without network access). `test_greeks_engine.py` and `test_scenario_analysis.py` check the aggregation/reporting layers rather than any pricing math directly. `test_visualization.py` is a smoke test confirming every chart function actually runs and returns a `Figure`. `test_model_validation.py` is my Phase 12 model validation suite: put-call parity, no-arbitrage bounds, American ≥ European, CRR/LR/Monte Carlo convergence, Greek stability under a small perturbation, and correct barrier behavior — this is the file I'd point a model validation reviewer at first.

## 4. A Note on My Finite Difference Barrier Pricer

My barrier PDE solver uses a first-order accurate explicit scheme on purpose, so the update rule stays a single readable line of code. This means a coarse grid can occasionally show a knock-out price a few cents above the true vanilla price (pure discretization error, not a modeling bug) — I discovered exactly this while testing at 80 price steps, and fixed it in my test suite by using a finer 150-step grid with a small numerical tolerance rather than chasing an exact bound. I document this trade-off explicitly in the "Limitations" section of `math_and_logic.tex` rather than hiding it.
