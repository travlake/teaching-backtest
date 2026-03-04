# Backtest Template Repository

## Purpose
Single-file backtesting framework for cross-sectional equity strategies using Compustat fundamentals and CRSP returns. Students clone this repo and modify `build_signal()` in `backtest_gp.py` to test a new trading signal. Everything else (portfolio formation, performance metrics, output) works automatically.

## File Structure
- `backtest_gp.py` — the backtest script (config + all functions in one file)

### Data files (not included in this repository)
The following data files are **not** included in the GitHub repository due to licensing restrictions. They are delivered to students via a zip file linked in the accompanying homework assignment. Place them in the same directory as `backtest_gp.py`.
- `compustat_with_permno.parquet` — Compustat quarterly fundamentals merged with CRSP PERMNO
- `crsp_m.dta` — CRSP monthly stock returns
- `ff5_plus_mom.dta` — Fama-French factors (monthly)

## How to Implement a New Strategy

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
| `atq` | Total assets, quarterly |
| `permno` | CRSP PERMNO (merged via CCM link) |
| `indfmt`, `consol`, `popsrc`, `datafmt`, `fic` | Standard Compustat filters |

### CRSP columns (monthly returns)
| Column | Description |
|--------|-------------|
| `PERMNO` | Permanent security identifier |
| `date` | Month-end date |
| `RET` | Monthly holding-period return (decimal) |
| `PRC` | Month-end price (negative = bid/ask midpoint) |
| `SHROUT` | Shares outstanding (thousands) |
| `SHRCD` | Share code (10, 11 = common US equity) |

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

4. **Staleness**: Signals become stale. The pipeline drops signals older than `STALENESS_DAYS` (default 365 days). This prevents using year-old fundamentals for firms that stopped reporting.

5. **Signal direction**: Higher signal values go into the long leg (top decile). If your signal is "bad" when high (e.g., leverage), negate it in `build_signal()`.

## Verification Checklist
After implementing a new signal:
- [ ] Script runs without errors
- [ ] Stock counts per month are reasonable (typically 1,000-4,000 in each decile)
- [ ] t-statistics are in a plausible range (|t| < 10 for most strategies)
- [ ] Sample period looks correct (check start/end dates)
- [ ] Long-short return has the expected sign
- [ ] Cumulative return plot looks reasonable (no vertical jumps suggesting data errors)

## Known Limitations
- No delisting return adjustment (uses CRSP monthly returns as-is)
- Breakpoints use all stocks, not NYSE-only
- No exchange code filter (includes NYSE, AMEX, NASDAQ)
- Single-factor CAPM alpha only (no FF3/FF5 alpha)
- Equal-weighted and value-weighted only (no other weighting schemes)
