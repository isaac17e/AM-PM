# AM-PM

A collection of Python scripts for **portfolio construction, active management, and risk monitoring**, built on market data (Yahoo Finance) and the options chain from **Polygon.io**.

The repository combines classical portfolio optimization (mean-variance, quadratic utility, minimum variance, Black-Litterman) with forward-looking information extracted from the options market: implied volatility, Bakshi-Kapadia-Madan (BKM) risk-neutral moments, dealer gamma exposure (GEX), order flow, and Cornish-Fisher expansions for tail risk.

> ⚠️ **Disclaimer**: this code is quantitative research and analysis material. It is not investment advice. The default parameters (portfolios, tickers, rates) are examples and should be adjusted before any real use.

---

## Table of contents

- [Workflow architecture](#workflow-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Scripts](#scripts)
  - [1. Portfolio construction](#1-portfolio-construction)
  - [2. Active management and entry signals](#2-active-management-and-entry-signals)
  - [3. Risk and visualization](#3-risk-and-visualization)
- [Key concepts](#key-concepts)
- [Notes on the Polygon API](#notes-on-the-polygon-api)

---

## Workflow architecture

The scripts run independently, but they are designed to chain together: the output of an optimizer (a `ticker: weight` dictionary) becomes the input for the management and risk modules.

```
   [ SELECTION + OPTIMIZATION ]
   quadratic_utility.py
   minimum_variance.py            ──►  portfolio  { "GLD": 0.18, "DHR": 0.18, ... }
   black_litterman.py                        │
                                             ▼
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                       [ ENTRY TIMING ]          [ MANAGEMENT AND RISK ]
                   entry_signal_tool.py           active_management.py
                                                  portfolio_risk_score_leverage.py
                                                  portfolio_gex_field.py
```

---

## Requirements

- Python 3.9 or higher
- A [Polygon.io](https://polygon.io) API key (the options modules will not work without one)

Libraries used:

| Area | Packages |
|---|---|
| Data and computation | `numpy`, `pandas`, `requests`, `python-dateutil` |
| Market data | `yfinance`, `pandas-datareader`, `beautifulsoup4` |
| Statistics / optimization | `scipy`, `statsmodels`, `quadprog` |
| Visualization | `plotly`, `matplotlib` |
| Environment | `python-dotenv` |

---

## Installation

```bash
git clone https://github.com/isaac17e/AM-PM.git
cd AM-PM

python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install numpy pandas requests python-dateutil \
            yfinance pandas-datareader beautifulsoup4 \
            scipy statsmodels quadprog \
            plotly matplotlib python-dotenv
```

`quadprog` requires a C compiler. On Windows it is usually easier to install it from a prebuilt wheel.

## Configuration

Create a `.env` file in the repository root (it is already covered by `.gitignore`):

```env
POLYGON_API_KEY=your_api_key_here
```

Every script loads it via `load_dotenv()`. If the key is missing, the optimizers print a warning and fall back to historical methods, while the options-driven modules (`black_litterman.py` with `USAR_IV_POLYGON = True`, `portfolio_gex_field.py`, `active_management.py`, `entry_signal_tool.py`) will not run correctly.

**Parameters are edited directly in the configuration block at the top of each file.** There is no CLI and no external config file: each script runs with `python script_name.py` and opens its Plotly charts in the browser.

---

## Scripts

### 1. Portfolio construction

#### `quadratic_utility.py` (~2,460 lines)
Portfolio optimizer based on **quadratic utility maximization** (`U = μ'w − λ/2 · w'Σw`).

Pipeline:
1. Builds the universe (top S&P 500 and NASDAQ names via scraping, commodities, ETFs, and international tickers), mapping currency by ticker suffix and converting prices to USD through FX pairs.
2. Computes descriptive statistics, correlations, and **Fama-French 3-factor betas** (`pandas_datareader`).
3. **Joint candidate selection via QUBO/Ising**: brute force when the search space is small, simulated annealing otherwise, with an objective blending Sharpe, low volatility, and decorrelation.
4. Applies sequential filters: delta, recent volatility, IV vs. realized volatility, and a **BKM MFIS filter** (flags anomalous hedging vs. speculation via a z-score against the asset's own historical MFIS).
5. Builds the covariance matrix with **shrinkage between implied (BKM) and historical covariance**, penalized by implied skewness and kurtosis (`theta1`, `theta2`).
6. Optimizes with `quadprog` and produces: efficient frontier, lambda comparison, risk attribution by Greeks, maximum drawdown analysis, and an executive summary.

Key parameters: `lambda_`, `max_weight`, `horizon_months`, `target_total_tickers`, `bkm_z_threshold`, `cornish_fisher_confidence`.

#### `minimum_variance.py` (~2,100 lines)
Optimizer for **minimum prospective tail risk (BKM + Cornish-Fisher)**, an evolution of the classical minimum-variance approach.

- Replaces the traditional volatility filter with a **Cornish-Fisher VaR** ranking built from risk-neutral moments.
- **Implied correlation model by factors**: market + sector + country + FX.
- Blends implied and historical covariance, penalizing the diagonal for kurtosis (`tail_risk_alpha`) and negative skewness (`tail_risk_beta`).
- Configurable constraints: max/min weight per asset, maximum number of holdings, ETF participation (`etf_max_weight`), and currency exposure (`max_fx_exposure`).
- Outputs: efficient frontier, Plotly visualizations, and synthetic risk attribution using Black-Scholes Greeks.

#### `black_litterman.py` (~1,100 lines)
**Black-Litterman** implementation over a manually defined ticker universe.

- Equilibrium returns `π` from reverse CAPM, using market capitalizations as reference weights.
- Implied volatility from Polygon with an **SSVI** surface fit and PCHIP interpolation.
- **BKM** module for higher-order risk-neutral moments, used as a bridge to systematically construct `Q` and `Ω` (the views and their uncertainty).
- **MVSK** optimization (mean-variance-skewness-kurtosis) instead of pure mean-variance.
- Predefined risk profiles (`conservador`, `moderado`, `agresivo`) that set `tau`, `omega_scale`, and `gamma_ra`.
- Includes maximum drawdown analysis of the resulting portfolio.

Manager views are edited in **Block 6** of the file.

#### Seasonal versions
`minimum_variance_(seasonal_version).py` and `quadratic_utility_(seasonal_version).py` mirror the pipelines above, but restrict the analysis to **specific months of the year** (`execution_months` / `rebalance_months`, defaulting to `[9]`).

Two things change:
- Return and risk statistics are computed over the **historical seasonal window** rather than the full series (`seasonal_min_weeks` sets the minimum number of valid observations).
- An exclusion filter on `seasonal_vol_ratio_max` is added: if an asset's seasonal volatility exceeds its general volatility by more than that multiple, it is dropped; `seasonal_min_survivors` prevents the universe from emptying out.

Use these when optimizing for a specific entry month rather than a generic horizon.

---

### 2. Active management and entry signals

#### `active_management.py` (~790 lines)
**Tactical rebalancing** engine for an existing portfolio, driven by options-market microstructure.

Blocks:
1. Options chain extraction and processing (Polygon v3, with pagination and retries).
2. Microstructure metrics: **GEX** (gamma exposure) and the zero-gamma flip point, order flow, unusual options activity (UOA), sweep dominance, put/call ratio, and **vanna-charm** effects.
3. A weighted **tactical score** (`score_gex_weight`, `score_flow_sweep`, `score_pcr_scale`, …) that maps to a per-asset recommendation.
4. Rebalancing with explicit thresholds: increase (score ≥ 50), hold, trim (≤ −50), with freed cash allocated up to the `cash_reserve_limit` cap (30% by default).
5. Reporting: gamma profiles consolidated onto a single page, plus a before/after allocation chart.

Input: the `portfolio` dictionary and `investment_horizon_days` at the top of the file.

#### `entry_signal_tool.py` (~310 lines)
A lightweight **entry-timing conviction score**. For each ticker it computes eight indicators and normalizes them by historical percentile:

| Indicator | Weight |
|---|---|
| GEX regime | 0.18 |
| Distance to zero gamma | 0.15 |
| IV rank | 0.15 |
| Room to the walls | 0.12 |
| Skew | 0.10 |
| Expected move | 0.10 |
| Smart money | 0.10 |
| Relative volume | 0.10 |

The resulting score (0-100) maps to a **suggested entry percentage**: full entry above 75, a graduated partial entry between 40 and 75, and a minimal entry below that.

Each run persists results to `entry_signal_history.csv` (one row per ticker per day, replacing the same day's row if it already exists). That history is what feeds the percentiles, so **the first few runs produce weak signals** until enough history accumulates.

It respects Polygon's free-tier limit by waiting `SEGUNDOS_ENTRE_LLAMADAS_STOCKS` (13 s) between tickers.

---

### 3. Risk and visualization

#### `portfolio_risk_score_leverage.py` (~990 lines)
A two-layer risk pipeline plus a leverage module:

- **Ex-post risk (historical)**: annualized volatility, historical VaR and CVaR at 95% and 99%, drawdowns, and risk-adjusted ratios (configurable MAR).
- **Ex-ante risk (options)**: ATM IV, GEX profile, and put/call ratio by open interest, sourced from Polygon.
- **Dynamic per-asset leverage**: a 0-100 `Risk_Score` weighting HV (30%), CVaR (30%), IV (25%), and the GEX/PCR block (15%), mapped inversely to a leverage range between `leverage_min` (2.0) and `leverage_max` (5.0).

Ends with a tabular report and per-asset charts.

#### `portfolio_gex_field.py` (~980 lines)
A 3D visualization of the portfolio as a **composite force field**.

- X axis: price in sigma units (±3σ, 60-point resolution).
- Y axis: composite macro factor (VIX 50%, TNX 25%, DXY 25%).
- Z axis: potential surface derived from smoothed net GEX, with macro amplification (`BETA_MACRO_AMPLIFICATION`) and tilt (`GAMMA_MACRO_TILT`).

Portfolio positions are plotted on the surface along with gradient vectors indicating the path of least resistance. Output is written to `portfolio_gex_field.html` with **auto-refresh every 60 seconds** (`REFRESH_SECONDS`), optional camera rotation, and automatic browser launch. This is the only script meant to be left running in a loop.

---

## Key concepts

- **BKM (Bakshi, Kapadia, and Madan, 2003)**: extraction of *risk-neutral* variance, skewness, and kurtosis (MFIV, MFIS, MFIK) by integrating OTM option prices. Used here as a forward-looking risk estimator, in contrast to historical moments.
- **Cornish-Fisher**: an expansion that adjusts normal quantiles for skewness and kurtosis, producing a VaR/CVaR sensitive to fat tails. The scripts clip the moments because the expansion breaks down under extreme tails.
- **GEX (Gamma Exposure)**: aggregate dealer gamma exposure. Positive GEX is associated with range-bound markets (dealers dampen moves); negative GEX with amplified moves. The zero-gamma flip marks the boundary between the two regimes.
- **Implied/historical shrinkage**: the final covariance is a weighted blend of the implied-volatility estimate and the historical one, with the weight controlled by `shrinkage_min`, `shrinkage_max`, and `ratio_band`.
- **QUBO/Ising**: asset selection is framed as a quadratic binary optimization problem, where the `h_i` terms capture individual quality and `J_ij` penalizes correlation.

---

## Notes on the Polygon API

- The free *Stocks Basic* tier allows **5 calls per minute**; several scripts include pauses and retries (`polygon_max_retries`, `polygon_retry_wait_sec`, `SEGUNDOS_ENTRE_LLAMADAS_STOCKS`) to stay within it.
- The optimizers iterate over hundreds of tickers, so on the free tier a full run can take hours. Lower `n_top_sp500`, `n_top_nasdaq`, and `target_total_tickers` for quick tests.
- `bkm_max_workers` controls parallelism in the BKM filter; raise it only if your plan allows.
- When an options query fails, the code falls back to a historical method (no real BKM) and reports it in the output. Always check the **data quality** section at the end of the optimizers.

---

## License

No license declared in the repository.
