# Backtest Template Repository

## Purpose
Backtesting framework for cross-sectional equity strategies using Compustat fundamentals and CRSP returns. Contains two backtest scripts:

- **`backtest_gp.py`** — Template for testing Compustat-based signals (monthly rebalancing, all-stock breakpoints). Students modify `build_signal()` to test new signals.
- **`backtest_bm_ff.py`** — Fama-French-style book-to-market strategy (annual June rebalancing, NYSE breakpoints, December ME). Replicates the value results shown in Lecture 16.

## File Structure
- `backtest_gp.py` — general-purpose backtest template (config + all functions in one file)
- `backtest_bm_ff.py` — B/M value strategy following Fama-French methodology

### Data files (not included in this repository)
Place data files in the same directory as the scripts. Two CRSP formats are supported:
- `compustat_with_permno.parquet` — Compustat quarterly fundamentals merged with CRSP PERMNO
- `crsp_m.parquet` — CRSP monthly stock returns (preferred; includes EXCHCD for NYSE breakpoints, delisting-adjusted returns)
- `crsp_m.dta` — CRSP monthly stock returns (legacy Stata format; works with `backtest_gp.py` but lacks EXCHCD)
- `ff5_plus_mom.dta` — Fama-French factors (monthly)

`backtest_gp.py` auto-detects whichever CRSP file is present (.parquet or .dta).

## How to Implement a New Strategy (backtest_gp.py)

### Step 1: Update the CONFIG block (top of `backtest_gp.py`)
```python
STRATEGY_NAME   = "Your Strategy Name"
SIGNAL_COL      = "your_signal"     # column name for your signal
LAG_MONTHS      = 4                 # adjust if your signal has different publication lag
```

### Step 2: Rewrite `build_signal(comp)`
This is the **only function you need to change**. It receives cleaned Compustat data and must return a DataFrame with exactly these columns:
| Column | Type | Description |
|--------|------|-------------|
| `PERMNO` | int | CRSP permanent security identifier |
| `signal_avail` | datetime | First month the signal can be used (start of month) |
| `<SIGNAL_COL>` | float | The signal value. Higher = long leg, lower = short leg |
| `datadate` | datetime | Fiscal quarter end date (used for staleness tracking) |

### Step 3: Run
```
python backtest_gp.py
```
Output: metrics table in console + CSV, TXT, and PNG files.

## B/M FF Backtest (backtest_bm_ff.py)

This script implements the classic Fama-French book-to-market value strategy:

- **Signal**: B/M = book equity (last FYE in calendar year t-1) / market cap (December t-1)
  - Book equity: Compustat `ceqq` with `atq - ltq` fallback; requires BE > 0
  - Market equity: CRSP price x shares in December
- **Formation**: end of June each year
  - Decile breakpoints from NYSE stocks only (EXCHCD == 1)
  - All stocks assigned to deciles using those breakpoints
- **Holding period**: July t through June t+1
  - Decile membership locked; stocks that delist drop out naturally
  - VW weights update monthly using beginning-of-month market cap
- **Requires**: `crsp_m.parquet` (needs EXCHCD column for NYSE breakpoints)

```
python backtest_bm_ff.py
```

## Data Dictionary

### Compustat columns (quarterly fundamentals)
| Column | Description |
|--------|-------------|
| `gvkey` | Compustat firm identifier |
| `datadate` | Fiscal quarter end date |
| `fyearq` | Fiscal year |
| `fqtr` | Fiscal quarter (1-4) |
| `revtq` | Revenue, total, quarterly |
| `saleq` | Sales/turnover, quarterly (fallback for revtq) |
| `cogsq` | Cost of goods sold, quarterly |
| `ceqq` | Common equity, quarterly (used in B/M) |
| `atq` | Total assets, quarterly |
| `ltq` | Total liabilities, quarterly (used in B/M fallback) |
| `permno` | CRSP PERMNO (merged via CCM link) |
| `indfmt`, `consol`, `popsrc`, `datafmt`, `fic` | Standard Compustat filters |

### CRSP columns (monthly returns)
| Column | Description |
|--------|-------------|
| `PERMNO` | Permanent security identifier |
| `date` | Month-end date |
| `RET` | Monthly holding-period return (decimal, delisting-adjusted in .parquet) |
| `PRC` | Month-end price (negative = bid/ask midpoint) |
| `SHROUT` | Shares outstanding (thousands) |
| `SHRCD` | Share code (10 = common US equity) |
| `EXCHCD` | Exchange code (1=NYSE, 2=AMEX, 3=NASDAQ) — .parquet only |

### Fama-French factor columns
| Column | Description |
|--------|-------------|
| `dateff` | Month-end date |
| `mktrf` | Market excess return |
| `rf` | Risk-free rate (monthly) |

## Common Pitfalls

1. **Look-ahead bias**: The `signal_avail` date must be *after* the data could realistically be known. Use `datadate + LAG_MONTHS` (typically 4 months for quarterly data). Never use `datadate` directly as the signal date.

2. **Long-short Rf subtraction**: Long-short is a zero-cost portfolio. In CAPM regression, regress L/S returns on MktRf directly (do NOT subtract Rf). For long and short legs individually, subtract Rf first. The `compute_metrics()` function handles this via the `is_long_short` flag.

3. **Lagged price filter**: Use *lagged* price (not current price) to avoid conditioning on future information. The `MIN_PRICE` config controls the cutoff (default $3).

4. **Staleness**: Signals become stale. The `backtest_gp.py` pipeline drops signals older than `STALENESS_DAYS` (default 365 days). This prevents using year-old fundamentals for firms that stopped reporting.

5. **Signal direction**: Higher signal values go into the long leg (top decile). If your signal is "bad" when high (e.g., leverage), negate it in `build_signal()`.

6. **NYSE breakpoints**: `backtest_bm_ff.py` uses NYSE-only breakpoints to avoid letting tiny NASDAQ stocks drive the decile cutoffs. This requires the `EXCHCD` column (only in `crsp_m.parquet`).

## Verification Checklist
After implementing a new signal:
- [ ] Script runs without errors
- [ ] Stock counts per month are reasonable (typically 1,000-4,000 in each decile)
- [ ] t-statistics are in a plausible range (|t| < 10 for most strategies)
- [ ] Sample period looks correct (check start/end dates)
- [ ] Long-short return has the expected sign
- [ ] Cumulative return plot looks reasonable (no vertical jumps suggesting data errors)
