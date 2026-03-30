# Backtest Template

A backtesting framework for cross-sectional equity strategies, built for students at **McCombs School of Business** at The University of Texas at Austin.

## Overview

This repository provides two ready-to-use backtesting scripts:

- **`backtest_gp.py`** — A general-purpose template for testing long-short trading signals based on accounting fundamentals. You describe the trading signal you want to test — in plain English — and your CLI coding agent (Claude Code or similar) handles all the code changes for you. No programming experience required.

- **`backtest_bm_ff.py`** — A Fama-French-style book-to-market value strategy that replicates the classic value premium results. Use this to understand the value factor, or as a reference for how annual-rebalancing strategies are implemented.

Both scripts handle portfolio formation (decile sorts), return computation (equal- and value-weighted), performance evaluation (CAPM alpha, Sharpe ratio, drawdowns), and output generation automatically.

## Getting Started

You'll interact with this project entirely through your CLI coding agent. The agent can read files, write code, install packages, and run the backtest for you — just tell it what you want in plain language.

### 1. Download and set up the project

Your homework assignment contains a link to a zip file. Download and unzip it — this gives you the data files needed to run the backtest. Then ask your agent:

> "Clone [repo-url] and move the data files into the project folder."

The data files you need (from the zip) are:

- `compustat_with_permno.parquet` — Compustat quarterly fundamentals
- `crsp_m.parquet` — CRSP monthly stock returns (preferred; includes exchange codes and delisting-adjusted returns)
- `ff5_plus_mom.dta` — Fama-French factors

If you have the older data delivery with `crsp_m.dta` instead, that's fine — `backtest_gp.py` auto-detects either format. (The FF-style script `backtest_bm_ff.py` requires the `.parquet` version for NYSE breakpoints.)

### 2. Install dependencies

Ask your agent:

> "Install the Python dependencies needed to run backtest_gp.py."

### 3. Replicate the Lecture 16 results

To see the classic B/M value strategy:

> "Run backtest_bm_ff.py and show me the results."

This produces performance metrics and cumulative return plots for the Fama-French value factor.

### 4. Implement your own signal

This is the core of the assignment. Describe the trading signal you want to test and ask your agent to implement it. For example:

> "I want to test a gross profitability signal. Compute gross profit as revenue minus cost of goods sold, scaled by total assets. Higher values should go in the long portfolio."

The agent will modify `backtest_gp.py` for you — specifically the `build_signal()` function and the configuration block at the top of the file. You don't need to understand the code; just make sure you can describe the signal clearly.

### 5. Run the backtest

Ask your agent:

> "Run the backtest."

Output includes:
- A performance metrics table (returns, alpha, Sharpe, drawdowns, t-statistics)
- CSV and TXT files with the metrics
- PNG cumulative return plots

### 6. Interpret and iterate

Review the output with your agent. You can ask things like:

> "What does the CAPM alpha tell us about this strategy?"
>
> "The long-short return is negative — does that mean I have the signal direction wrong?"
>
> "Can you flip the signal so that low values go in the long portfolio?"

## What You'll Learn

- How to construct and evaluate long-short equity portfolios
- The importance of avoiding look-ahead bias (publication lag, lagged price filters)
- Decile portfolio sorts and cross-sectional signal analysis
- CAPM alpha estimation and statistical inference on portfolio returns
- The difference between equal-weighted and value-weighted portfolio returns
- How methodological choices (breakpoints, rebalancing frequency, exchange filters) affect results

## Tips for Working with Your Agent

- **Be specific about your signal.** Describe exactly which accounting variables to use and how to combine them. The clearer you are, the better the result.
- **Ask it to explain.** If you're curious about what the code does or what a metric means, just ask. The agent can walk you through it.
- **Iterate.** If results look wrong or surprising, ask the agent to investigate. It can check stock counts, look at summary statistics, or adjust the signal.
- **Check the verification list.** Ask your agent: *"Run through the verification checklist in CLAUDE.md and tell me if anything looks off."*

## Common Pitfalls

- **Look-ahead bias**: Signals must not use information before it would realistically be available. The framework enforces a publication lag (default: 4 months after quarter end).
- **Signal direction**: Higher signal values go into the long leg (top decile). If your signal is "bad" when high (e.g., leverage), tell your agent to negate it.
- **Staleness**: Signals older than 365 days are automatically dropped to prevent using outdated fundamentals.

## License

For academic use at McCombs School of Business. Data files are subject to CRSP and Compustat licensing terms and must not be redistributed.
