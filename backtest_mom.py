"""
Backtest: Momentum (Prior 2-12 Month Returns)
==============================================
Signal : cumulative return from month t-12 to t-2 (skip most recent month)
Long   : top decile momentum (winners)
Short  : bottom decile momentum (losers)
Rebal  : monthly
Universe: common US equities (SHRCD 10/11), lagged price > $3
Breaks : NYSE-only breakpoints, all stocks in portfolios
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats
import warnings, time

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG
# =====================================================================
STRATEGY_NAME   = "Momentum (2-12)"
SIGNAL_COL      = "mom"
DATA_DIR        = Path(__file__).resolve().parent
CRSP_FILE       = "crsp_m.parquet"
FF_FILE         = "ff5_plus_mom.dta"
MIN_PRICE       = 3
N_QUANTILES     = 10
PLOT_DPI        = 200
MOM_START       = 12       # start of lookback window (months ago)
MOM_END         = 2        # end of lookback window (months ago); skips most recent month


# =====================================================================
# LOAD DATA
# =====================================================================
def load_data():
    """Load CRSP monthly and Fama-French factors."""
    crsp = pd.read_parquet(DATA_DIR / CRSP_FILE, engine="fastparquet")
    ff   = pd.read_stata(DATA_DIR / FF_FILE, columns=["dateff", "mktrf", "rf"])
    return crsp, ff


# =====================================================================
# CLEAN CRSP
# =====================================================================
def clean_crsp(crsp):
    """Filter CRSP, compute lagged price/mktcap, keep EXCHCD."""
    crsp.columns = crsp.columns.str.upper()
    crsp.rename(columns={"DATE": "date"}, inplace=True)
    crsp["date"]    = pd.to_datetime(crsp["date"])
    crsp            = crsp[crsp["SHRCD"].isin([10, 11])].copy()
    crsp.dropna(subset=["RET"], inplace=True)
    crsp["abs_prc"] = crsp["PRC"].abs()
    crsp["mktcap"]  = crsp["abs_prc"] * crsp["SHROUT"]
    crsp["PERMNO"]  = crsp["PERMNO"].astype(int)
    crsp["month"]   = crsp["date"].dt.to_period("M").dt.to_timestamp()

    crsp.sort_values(["PERMNO", "month"], inplace=True)
    crsp["lag_prc"]    = crsp.groupby("PERMNO")["abs_prc"].shift(1)
    crsp["lag_mktcap"] = crsp.groupby("PERMNO")["mktcap"].shift(1)
    crsp = crsp[crsp["lag_prc"] > MIN_PRICE].copy()

    if "EXCHCD" in crsp.columns:
        crsp["EXCHCD"] = crsp["EXCHCD"].astype(int)
    return crsp


# =====================================================================
# BUILD MOMENTUM SIGNAL
# =====================================================================
def build_signal(crsp):
    """
    Compute prior 2-12 month cumulative return for each stock-month.

    For month t, the signal is the cumulative return from t-12 to t-2
    (i.e., skip the most recent month t-1 to avoid short-term reversal).

    Requires at least (MOM_START - MOM_END + 1) = 11 valid returns in
    the lookback window.
    """
    crsp = crsp.sort_values(["PERMNO", "month"]).copy()

    # Compute cumulative return over rolling window
    # cum_ret_12 = cumulative return from t-12 to t-1 (12 months)
    # cum_ret_1  = return in t-1 (most recent month)
    # momentum   = (1 + cum_ret_12) / (1 + cum_ret_1) - 1
    n_months = MOM_START  # need 12 months of returns

    crsp["log_ret"] = np.log(1 + crsp["RET"])
    crsp["cum_log_12"] = crsp.groupby("PERMNO")["log_ret"].transform(
        lambda x: x.shift(1).rolling(n_months, min_periods=n_months).sum()
    )
    crsp["log_ret_1"] = crsp.groupby("PERMNO")["log_ret"].shift(1)

    # Prior 2-12: subtract most recent month from the full 12-month window
    crsp["cum_log_2_12"] = crsp["cum_log_12"] - crsp["log_ret_1"]
    crsp[SIGNAL_COL] = np.exp(crsp["cum_log_2_12"]) - 1

    crsp.dropna(subset=[SIGNAL_COL], inplace=True)
    crsp.drop(columns=["log_ret", "cum_log_12", "log_ret_1", "cum_log_2_12"],
              inplace=True)
    return crsp


# =====================================================================
# ASSIGN PORTFOLIOS (monthly, NYSE breakpoints)
# =====================================================================
def assign_portfolios(crsp):
    """
    Each month, assign stocks to deciles using NYSE momentum breakpoints.
    """
    crsp = crsp.copy()
    crsp.dropna(subset=["RET", SIGNAL_COL, "lag_mktcap"], inplace=True)

    def _assign_decile(group):
        nyse = group[group["EXCHCD"] == 1][SIGNAL_COL]
        if len(nyse) < N_QUANTILES:
            group["decile"] = np.nan
            return group
        breaks = np.percentile(nyse, np.linspace(0, 100, N_QUANTILES + 1))
        breaks[0] = -np.inf
        breaks[-1] = np.inf
        group["decile"] = pd.cut(
            group[SIGNAL_COL], bins=breaks, labels=False, include_lowest=True
        ) + 1
        return group

    merged = crsp.groupby("month", group_keys=False).apply(_assign_decile)
    merged.dropna(subset=["decile"], inplace=True)
    merged["decile"] = merged["decile"].astype(int)
    merged["port"] = np.where(merged["decile"] == N_QUANTILES, "long",
                     np.where(merged["decile"] == 1, "short", "mid"))
    return merged


# =====================================================================
# PORTFOLIO RETURNS
# =====================================================================
def compute_portfolio_returns(merged):
    """Compute EW and VW returns for long, short, and long-short."""
    ew = merged.groupby(["month", "port"])["RET"].mean().unstack("port")
    merged["wt_ret"] = merged["lag_mktcap"] * merged["RET"]
    vw_agg = (merged.groupby(["month", "port"])[["wt_ret", "lag_mktcap"]].sum())
    vw_agg["vw"] = vw_agg["wt_ret"] / vw_agg["lag_mktcap"]
    vw = vw_agg["vw"].unstack("port")
    merged.drop(columns="wt_ret", inplace=True)

    ew["long_short"] = ew["long"] - ew["short"]
    vw["long_short"] = vw["long"] - vw["short"]

    results = pd.DataFrame(index=ew.index)
    results.index.name = "date"
    for pfx, src in [("ew", ew), ("vw", vw)]:
        for col in ["long", "short", "long_short"]:
            results[f"{pfx}_{col}"] = src[col]
    results.dropna(how="all", inplace=True)
    return results


# =====================================================================
# PERFORMANCE METRICS
# =====================================================================
def compute_metrics(r, name, ff_df, is_long_short=False):
    """Full metrics suite aligned with Lecture 12."""
    r = r.dropna()
    n = len(r)
    mu      = r.mean()
    sigma   = r.std(ddof=1)
    ann_ret = mu * 12
    ann_vol = sigma * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    se_mu   = sigma / np.sqrt(n)
    t_mu    = mu / se_mu if se_mu > 0 else np.nan
    p_mu    = 2 * (1 - scipy_stats.t.cdf(abs(t_mu), df=n-1))
    ci_lo   = ann_ret - 1.96 * sigma * np.sqrt(12 / n)
    ci_hi   = ann_ret + 1.96 * sigma * np.sqrt(12 / n)
    geo     = (1 + r).prod() ** (12 / n) - 1
    cum     = (1 + r).cumprod()
    drawdowns = (cum - cum.cummax()) / cum.cummax()
    max_dd  = drawdowns.min()
    avg_dd  = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0

    # CAPM regression
    df = pd.DataFrame({"ret": r}).reset_index()
    df.columns = ["month", "ret"]
    df = df.merge(ff_df, on="month", how="inner")
    df["y"] = df["ret"] if is_long_short else df["ret"] - df["rf"]
    X = sm.add_constant(df["mktrf"])
    capm = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    alpha_m = capm.params["const"]
    beta    = capm.params["mktrf"]
    alpha_a = alpha_m * 12
    alpha_t = capm.tvalues["const"]
    alpha_p = capm.pvalues["const"]
    alpha_se_a = capm.bse["const"] * 12
    alpha_ci_lo = alpha_a - 1.96 * alpha_se_a
    alpha_ci_hi = alpha_a + 1.96 * alpha_se_a

    return {
        "name": name,
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
        "months": n,
        "ann_ret": ann_ret,
        "geo_ret": geo,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "t_stat_ret": t_mu,
        "p_val_ret": p_mu,
        "ret_ci_lo": ci_lo,
        "ret_ci_hi": ci_hi,
        "avg_dd": avg_dd,
        "max_dd": max_dd,
        "capm_alpha": alpha_a,
        "alpha_se": alpha_se_a,
        "alpha_t": alpha_t,
        "alpha_p": alpha_p,
        "alpha_ci_lo": alpha_ci_lo,
        "alpha_ci_hi": alpha_ci_hi,
        "capm_beta": beta,
        "beta_t": capm.tvalues["mktrf"],
    }


# =====================================================================
# OUTPUT (console, CSV, TXT)
# =====================================================================
def output_results(results, metrics):
    """Print formatted tables, save CSV/TXT."""
    def fmt_table(m):
        out = pd.DataFrame(index=m.index)
        out["Period"]        = m["start"] + " to " + m["end"]
        out["Months"]        = m["months"].astype(int)
        out["Arith Mean"]    = m["ann_ret"].map(lambda x: f"{x:.2%}")
        out["Geo Mean"]      = m["geo_ret"].map(lambda x: f"{x:.2%}")
        out["Sigma"]         = m["ann_vol"].map(lambda x: f"{x:.2%}")
        out["Sharpe"]        = m["sharpe"].map(lambda x: f"{x:.2f}")
        out["t(mean)"]       = m["t_stat_ret"].map(lambda x: f"{x:.2f}")
        out["p(mean)"]       = m["p_val_ret"].map(lambda x: f"{x:.4f}")
        out["95% CI (ret)"]  = [f"[{lo:.2%}, {hi:.2%}]"
                                for lo, hi in zip(m["ret_ci_lo"], m["ret_ci_hi"])]
        out["Avg DD"]        = m["avg_dd"].map(lambda x: f"{x:.2%}")
        out["Max DD"]        = m["max_dd"].map(lambda x: f"{x:.2%}")
        out["CAPM Alpha"]    = m["capm_alpha"].map(lambda x: f"{x:.2%}")
        out["Alpha SE"]      = m["alpha_se"].map(lambda x: f"{x:.2%}")
        out["t(alpha)"]      = m["alpha_t"].map(lambda x: f"{x:.2f}")
        out["p(alpha)"]      = m["alpha_p"].map(lambda x: f"{x:.4f}")
        out["95% CI (alpha)"]= [f"[{lo:.2%}, {hi:.2%}]"
                                for lo, hi in zip(m["alpha_ci_lo"], m["alpha_ci_hi"])]
        out["CAPM Beta"]     = m["capm_beta"].map(lambda x: f"{x:.3f}")
        out["t(beta)"]       = m["beta_t"].map(lambda x: f"{x:.2f}")
        return out

    display = fmt_table(metrics)

    sep = "=" * 100
    header = (f"\n{sep}\n"
              f"{STRATEGY_NAME.upper()} BACKTEST\n"
              f"Signal: cumulative return months t-{MOM_START} to t-{MOM_END}   |   "
              f"Rebal: monthly\n"
              f"Universe: SHRCD 10/11, lagged |PRC| > ${MIN_PRICE}   |   "
              f"Breakpoints: NYSE {N_QUANTILES}-tiles\n"
              f"{sep}")

    print(header)
    for label in ["EW Long-Short", "VW Long-Short",
                  "EW Long", "EW Short", "VW Long", "VW Short"]:
        row = display.loc[label]
        print(f"\n--- {label} ---")
        print(row.to_string())
    print(f"\n{sep}")

    # --- Save files ---
    csv_path     = DATA_DIR / f"{SIGNAL_COL}_backtest_returns.csv"
    metrics_path = DATA_DIR / f"{SIGNAL_COL}_backtest_metrics.csv"
    txt_path     = DATA_DIR / f"{SIGNAL_COL}_backtest_metrics.txt"

    results.to_csv(csv_path)
    metrics.to_csv(metrics_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        f.write(display.T.to_string())
        f.write(f"\n\n{sep}\n")

    print(f"\nReturns CSV : {csv_path}")
    print(f"Metrics CSV : {metrics_path}")
    print(f"Metrics TXT : {txt_path}")


# =====================================================================
# PLOTTING
# =====================================================================
def plot_cumulative(results, pfx, title, path, start=None, end=None):
    """Single-panel cumulative return plot (2.5:1 aspect ratio)."""
    fig, ax = plt.subplots(figsize=(12.5, 5))

    sub = results.loc[start:end] if start or end else results

    all_dates = []
    for col, color, ls, lw, lbl in [
        (f"{pfx}_long",       "steelblue", "--",  1.4, "Winners (Top 10%)"),
        (f"{pfx}_short",      "firebrick", ":",   1.4, "Losers (Bottom 10%)"),
        (f"{pfx}_long_short", "black",     "-",   1.8, "Winners$-$Losers"),
    ]:
        s = sub[col].dropna()
        cum = (1 + s).cumprod()
        ax.plot(cum.index, cum.values, color=color, linestyle=ls,
                linewidth=lw, label=lbl)
        all_dates.extend([cum.index.min(), cum.index.max()])

    ax.margins(x=0)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Cumulative Return (log scale)")
    ax.set_xlabel("")
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    t0 = time.time()

    print("Loading data ...")
    crsp, ff = load_data()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("Cleaning CRSP ...")
    crsp = clean_crsp(crsp)

    print("Building momentum signal (prior 2-12 month returns) ...")
    crsp = build_signal(crsp)
    print(f"  {crsp[SIGNAL_COL].notna().sum():,} stock-month signals")

    print("Assigning monthly portfolios (NYSE breakpoints) ...")
    merged = assign_portfolios(crsp)
    print(f"  {len(merged):,} obs | "
          f"{merged['PERMNO'].nunique():,} stocks | "
          f"{merged['month'].nunique()} months")

    print("Computing portfolio returns ...")
    results = compute_portfolio_returns(merged)

    # Prepare FF factors
    ff["month"] = ff["dateff"].dt.to_period("M").dt.to_timestamp()
    ff = ff[["month", "mktrf", "rf"]].drop_duplicates("month")

    print("Running CAPM regressions ...")
    series_list = [
        ("ew_long",       "EW Long",       False),
        ("ew_short",      "EW Short",      False),
        ("ew_long_short", "EW Long-Short", True),
        ("vw_long",       "VW Long",       False),
        ("vw_short",      "VW Short",      False),
        ("vw_long_short", "VW Long-Short", True),
    ]
    metrics = pd.DataFrame(
        [compute_metrics(results[c], n, ff, is_long_short=ls)
         for c, n, ls in series_list]
    ).set_index("name")

    output_results(results, metrics)

    # --- Slide-ready plots ---
    print("\nGenerating plots ...")
    plot_cumulative(results, "ew",
                    "Momentum Strategy — Equal-Weighted (Full Sample)",
                    DATA_DIR / f"{SIGNAL_COL}_ew_full.png")
    plot_cumulative(results, "vw",
                    "Momentum Strategy — Value-Weighted (Full Sample)",
                    DATA_DIR / f"{SIGNAL_COL}_vw_full.png")
    plot_cumulative(results, "vw",
                    "Momentum Strategy — Value-Weighted (2010–2025)",
                    DATA_DIR / f"{SIGNAL_COL}_vw_2010_2025.png",
                    start="2010-01-01", end="2025-12-31")

    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
