"""
Backtest: Gross Profitability (GP) Strategy
============================================
Signal : GP = sum(rev - cogsq, trailing 4Q) / atq
Long   : top decile GP
Short  : bottom decile GP
Lag    : fiscal quarter end + 4 months
Universe: common US equities (SHRCD 10/11), lagged price > $3
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
# CONFIG — edit these to change strategy parameters
# =====================================================================
STRATEGY_NAME   = "Gross Profitability"
SIGNAL_COL      = "gp"                 # column name for the signal
DATA_DIR        = Path(__file__).resolve().parent
COMP_FILE       = "compustat_with_permno.parquet"
CRSP_FILE       = "crsp_m.parquet"      # also supports crsp_m.dta (auto-detected)
FF_FILE         = "ff5_plus_mom.dta"
LAG_MONTHS      = 4          # months after fiscal quarter end before signal is usable
MIN_PRICE       = 3          # lagged price filter ($)
N_QUANTILES     = 10         # number of portfolio bins (10 = deciles)
STALENESS_DAYS  = 365        # drop signal if older than this
TRANSACTION_COST_BP = 0      # one-way transaction cost in basis points (0 = no cost)


# =====================================================================
# LOAD DATA
# =====================================================================
def load_data():
    """Load Compustat, CRSP monthly, and Fama-French factors.

    CRSP: tries CRSP_FILE first (.parquet or .dta); if not found, falls back
    to the other format so the same script works with either data delivery.
    """
    comp = pd.read_parquet(
        DATA_DIR / COMP_FILE,
        engine="fastparquet",
        columns=["gvkey", "datadate", "fyearq", "fqtr",
                 "indfmt", "consol", "popsrc", "datafmt",
                 "fic", "revtq", "saleq", "cogsq", "atq", "permno"],
    )
    crsp_path = DATA_DIR / CRSP_FILE
    if not crsp_path.exists():
        # Fall back to the other format
        alt = crsp_path.with_suffix(".dta" if crsp_path.suffix == ".parquet"
                                    else ".parquet")
        if alt.exists():
            crsp_path = alt
    if crsp_path.suffix == ".parquet":
        crsp = pd.read_parquet(crsp_path, engine="fastparquet")
    else:
        crsp = pd.read_stata(
            crsp_path,
            columns=["PERMNO", "date", "RET", "PRC", "SHROUT", "SHRCD"],
        )
    ff   = pd.read_stata(DATA_DIR / FF_FILE,
                         columns=["dateff", "mktrf", "rf"])
    return comp, crsp, ff


# =====================================================================
# BUILD SIGNAL — *** THIS IS THE FUNCTION STUDENTS SHOULD MODIFY ***
# =====================================================================
def build_signal(comp):
    """
    Build the trading signal from Compustat data.

    STUDENTS: rewrite this function to implement your own signal.
    Must return a DataFrame with columns: [PERMNO, signal_avail, <SIGNAL_COL>]
      - PERMNO:       int, CRSP permanent security identifier
      - signal_avail: datetime, first month the signal can be used (start of month)
      - <SIGNAL_COL>: float, the signal value (higher = more desirable for long leg)

    The current implementation computes Gross Profitability:
        GP = sum(revtq - cogsq, trailing 4 quarters) / atq
    """
    # Standard Compustat filters
    comp = comp[
        (comp["indfmt"] == "INDL") & (comp["consol"] == "C") &
        (comp["popsrc"] == "D")    & (comp["datafmt"] == "STD") &
        (comp["fic"] == "USA")
    ].copy()

    comp["datadate"] = pd.to_datetime(comp["datadate"])
    comp["permno"]   = comp["permno"].astype("Int64")
    comp["rev"]      = comp["revtq"].fillna(comp["saleq"])          # fallback

    comp.dropna(subset=["permno", "rev", "cogsq", "atq"], inplace=True)
    comp = comp[comp["atq"] > 0]
    comp["gpq"] = comp["rev"] - comp["cogsq"]

    comp.sort_values(["gvkey", "datadate"], inplace=True)
    comp.drop_duplicates(subset=["gvkey", "datadate"], keep="last", inplace=True)

    comp["gpq_ttm"] = comp.groupby("gvkey")["gpq"].transform(
        lambda x: x.rolling(4, min_periods=4).sum()
    )
    comp.dropna(subset=["gpq_ttm"], inplace=True)
    comp[SIGNAL_COL] = comp["gpq_ttm"] / comp["atq"]

    # Signal available date = datadate + LAG_MONTHS (start of month)
    comp["signal_avail"] = (
        (comp["datadate"] + pd.DateOffset(months=LAG_MONTHS))
        .dt.to_period("M").dt.to_timestamp()
    )

    # Return only the columns the rest of the pipeline needs
    out = comp[["permno", "signal_avail", SIGNAL_COL, "datadate"]].copy()
    out.rename(columns={"permno": "PERMNO"}, inplace=True)
    out["PERMNO"] = out["PERMNO"].astype(int)
    return out


# =====================================================================
# RESAMPLE SIGNAL TO MONTHLY GRID
# =====================================================================
def resample_signal(sig):
    """Expand quarterly signal to monthly, forward-fill, drop stale."""
    sig.sort_values(["PERMNO", "signal_avail", "datadate"], inplace=True)

    def _resample(df):
        return (df.set_index("signal_avail")[[SIGNAL_COL]]
                  .resample("MS").last().ffill())

    signal = (sig.groupby("PERMNO", group_keys=True)
                 .apply(_resample).reset_index())
    signal.rename(columns={"signal_avail": "month"}, inplace=True)

    # Staleness: drop if signal origin > STALENESS_DAYS ago
    sig_dates = (sig.drop_duplicates(subset=["PERMNO", "signal_avail"], keep="last")
                    [["PERMNO", "signal_avail"]].copy())
    sig_dates.rename(columns={"signal_avail": "month"}, inplace=True)
    sig_dates["sig_origin"] = sig_dates["month"]
    signal = signal.merge(sig_dates, on=["PERMNO", "month"], how="left")
    signal["sig_origin"] = signal.groupby("PERMNO")["sig_origin"].ffill()
    signal = signal[
        (signal["month"] - signal["sig_origin"]) <= pd.Timedelta(days=STALENESS_DAYS)
    ].drop(columns="sig_origin")
    signal.dropna(subset=[SIGNAL_COL], inplace=True)
    return signal


# =====================================================================
# CLEAN CRSP
# =====================================================================
def clean_crsp(crsp):
    """Filter CRSP to common US equities, compute lagged price/mktcap."""
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
    return crsp


# =====================================================================
# MERGE & FORM PORTFOLIOS
# =====================================================================
def merge_and_form_portfolios(crsp, signal):
    """Merge CRSP with signal, assign quantile portfolios."""
    merged = crsp.merge(signal, on=["PERMNO", "month"], how="inner")
    merged.dropna(subset=["RET", SIGNAL_COL, "lag_mktcap"], inplace=True)

    merged["decile"] = merged.groupby("month")[SIGNAL_COL].transform(
        lambda x: pd.qcut(x, N_QUANTILES, labels=False, duplicates="drop") + 1
    )
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

    # --- Turnover & transaction costs ---
    turnover_long = pd.Series(dtype=float)
    turnover_short = pd.Series(dtype=float)
    if TRANSACTION_COST_BP > 0:
        months = sorted(merged["month"].unique())
        prev_long = set()
        prev_short = set()
        to_long = {}
        to_short = {}
        for m in months:
            cur = merged[merged["month"] == m]
            cur_long = set(cur.loc[cur["port"] == "long", "PERMNO"])
            cur_short = set(cur.loc[cur["port"] == "short", "PERMNO"])
            if prev_long:
                entered = len(cur_long - prev_long)
                exited = len(prev_long - cur_long)
                avg_size = (len(cur_long) + len(prev_long)) / 2
                to_long[m] = (entered + exited) / (2 * avg_size) if avg_size > 0 else 0
            else:
                to_long[m] = 0  # first month: no turnover
            if prev_short:
                entered = len(cur_short - prev_short)
                exited = len(prev_short - cur_short)
                avg_size = (len(cur_short) + len(prev_short)) / 2
                to_short[m] = (entered + exited) / (2 * avg_size) if avg_size > 0 else 0
            else:
                to_short[m] = 0
            prev_long = cur_long
            prev_short = cur_short
        turnover_long = pd.Series(to_long)
        turnover_short = pd.Series(to_short)
        # Cost = turnover * 2 (buy+sell) * cost_per_trade
        cost_per_trade = TRANSACTION_COST_BP / 10_000
        cost_long = turnover_long * 2 * cost_per_trade
        cost_short = turnover_short * 2 * cost_per_trade

    ew["long_short"] = ew["long"] - ew["short"]
    vw["long_short"] = vw["long"] - vw["short"]

    # Apply transaction costs (same turnover for EW and VW since it's name-based)
    # Each leg's return is reduced by its own turnover cost.
    # Long-short = (long - cost_long) - (short + cost_short)
    #   The short-seller pays cost_short on top of funding the short position.
    if TRANSACTION_COST_BP > 0:
        cl = cost_long.reindex(ew.index).fillna(0)
        cs = cost_short.reindex(ew.index).fillna(0)
        for src in [ew, vw]:
            src["long"] = src["long"] - cl
            src["short"] = src["short"]       # short-leg stock return unchanged
            src["long_short"] = src["long_short"] - cl - cs

    results = pd.DataFrame(index=ew.index)
    results.index.name = "date"
    for pfx, src in [("ew", ew), ("vw", vw)]:
        for col in ["long", "short", "long_short"]:
            results[f"{pfx}_{col}"] = src[col]
    results.dropna(how="all", inplace=True)

    # Attach turnover info to results for output
    if TRANSACTION_COST_BP > 0:
        results.attrs["turnover_long"] = turnover_long.reindex(results.index).fillna(0)
        results.attrs["turnover_short"] = turnover_short.reindex(results.index).fillna(0)
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
    geo     = (1 + r).prod() ** (12 / n) - 1                    # geometric mean
    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()

    # CAPM regression
    # Long-short is already zero-cost: regress R_LS on MktRf directly
    # Long/short legs are invested: regress R - Rf on MktRf
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
# OUTPUT (console, CSV, TXT, plot)
# =====================================================================
def output_results(results, metrics):
    """Print formatted tables, save CSV/TXT, generate plot."""
    # --- Formatted table ---
    def fmt_table(m):
        out = pd.DataFrame(index=m.index)
        out["Period"]        = m["start"] + " to " + m["end"]
        out["Months"]        = m["months"].astype(int)
        out["Arith Mean"]    = m["ann_ret"].map(lambda x: f"{x:.2%}")
        out["Geo Mean"]      = m["geo_ret"].map(lambda x: f"{x:.2%}")
        out["Volatility"]    = m["ann_vol"].map(lambda x: f"{x:.2%}")
        out["Sharpe"]        = m["sharpe"].map(lambda x: f"{x:.2f}")
        out["t(mean)"]       = m["t_stat_ret"].map(lambda x: f"{x:.2f}")
        out["p(mean)"]       = m["p_val_ret"].map(lambda x: f"{x:.4f}")
        out["95% CI (ret)"]  = [f"[{lo:.2%}, {hi:.2%}]"
                                for lo, hi in zip(m["ret_ci_lo"], m["ret_ci_hi"])]
        out["Max DD"]        = m["max_dd"].map(lambda x: f"{x:.2%}")
        out["CAPM alpha"]    = m["capm_alpha"].map(lambda x: f"{x:.2%}")
        out["alpha SE"]      = m["alpha_se"].map(lambda x: f"{x:.2%}")
        out["t(alpha)"]      = m["alpha_t"].map(lambda x: f"{x:.2f}")
        out["p(alpha)"]      = m["alpha_p"].map(lambda x: f"{x:.4f}")
        out["95% CI (alpha)"]= [f"[{lo:.2%}, {hi:.2%}]"
                                for lo, hi in zip(m["alpha_ci_lo"], m["alpha_ci_hi"])]
        out["CAPM beta"]     = m["capm_beta"].map(lambda x: f"{x:.3f}")
        out["t(beta)"]       = m["beta_t"].map(lambda x: f"{x:.2f}")
        return out

    display = fmt_table(metrics)

    sep = "=" * 100
    tc_line = ""
    if TRANSACTION_COST_BP > 0 and "turnover_long" in results.attrs:
        avg_to_l = results.attrs["turnover_long"].mean()
        avg_to_s = results.attrs["turnover_short"].mean()
        cost_per_trade = TRANSACTION_COST_BP / 10_000
        drag_l = avg_to_l * 2 * cost_per_trade * 12
        drag_s = avg_to_s * 2 * cost_per_trade * 12
        drag_ls = drag_l + drag_s
        tc_line = (f"Transaction cost: {TRANSACTION_COST_BP} bp one-way   |   "
                   f"Avg monthly turnover: Long {avg_to_l:.1%}, Short {avg_to_s:.1%}   |   "
                   f"Ann. cost drag: L {drag_l:.2%}, S {drag_s:.2%}, L/S {drag_ls:.2%}\n")
    header = (f"\n{sep}\n"
              f"{STRATEGY_NAME.upper()} BACKTEST\n"
              f"Signal: {SIGNAL_COL}   |   "
              f"Lag: datadate + {LAG_MONTHS} months\n"
              f"Universe: SHRCD 10/11, lagged |PRC| > ${MIN_PRICE}   |   "
              f"Breakpoints: all-stock {N_QUANTILES}-tiles   |   Rebalancing: monthly\n"
              f"{tc_line}"
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

    # --- Plot ---
    print("\nPlotting cumulative returns ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pfx, title in [
        (axes[0], "ew", "Equal-Weighted"),
        (axes[1], "vw", "Value-Weighted"),
    ]:
        for col, color, lbl in [
            (f"{pfx}_long",       "steelblue", f"Long (D{N_QUANTILES})"),
            (f"{pfx}_short",      "firebrick", "Short (D1)"),
            (f"{pfx}_long_short", "black",     "Long - Short"),
        ]:
            cum = (1 + results[col].dropna()).cumprod()
            ax.plot(cum.index, cum.values, color=color, linewidth=1.2, label=lbl)

        ax.set_yscale("log")
        ax.set_title(f"{STRATEGY_NAME} Strategy ({title})", fontsize=12)
        ax.set_ylabel("Cumulative Return (log scale, $1 invested)")
        ax.set_xlabel("")
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    plot_path = DATA_DIR / f"{SIGNAL_COL}_backtest.png"
    fig.savefig(plot_path, dpi=150)
    print(f"  Plot saved: {plot_path}")

    print(f"\nReturns CSV : {csv_path}")
    print(f"Metrics CSV : {metrics_path}")
    print(f"Metrics TXT : {txt_path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    t0 = time.time()

    print("Loading data ...")
    comp, crsp, ff = load_data()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("Building signal ...")
    sig = build_signal(comp)
    print(f"  {len(sig):,} firm-quarter signals")

    print("Resampling signals to monthly grid ...")
    signal = resample_signal(sig)
    print(f"  {len(signal):,} permno-month rows")

    print("Cleaning CRSP ...")
    crsp = clean_crsp(crsp)

    print("Merging & forming portfolios ...")
    merged = merge_and_form_portfolios(crsp, signal)
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
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
