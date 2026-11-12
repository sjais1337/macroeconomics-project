"""
Hormuz 2026: Real Output Response and Monetary Policy Benchmark
ECO331 Project - IIT Kanpur

This script reproduces the real economy and monetary policy analysis:
1. UNIDO manufacturing production (IIP) growth across exposure groups
2. Pre-war vs war output DiD and robustness to base-year choice
3. Central bank policy rate adjustments evaluated against the Taylor rule
4. Central bank "look-through" behavior (Headline vs Core Taylor rule)
"""

import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Visual styling
NAVY = "#1f4e79"
ACCENT = "#b85c38"
SAGE = "#5b7c99"
SLATE = "#5a6b7c"
INK = "#1c2833"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#d0d7de",
    "axes.linewidth": 0.8,
    "grid.color": "#eaeef2",
    "grid.linewidth": 0.7,
})


def load_data():
    exposure = pd.read_csv(DATA_DIR / "exposure.csv").dropna(subset=["exposure"]).copy()
    exposure = exposure[exposure["iso3"] != "EA20"].copy()
    exposure["exposure_pct"] = exposure["exposure"] * 100
    median_exp = exposure["exposure"].median()
    exposure["high_exposure"] = (exposure["exposure"] >= median_exp).astype(int)

    unido = pd.read_csv(DATA_DIR / "unido_iip_monthly.csv", parse_dates=["date"])
    taylor = pd.read_csv(DATA_DIR / "taylor_by_bank.csv")

    return exposure, unido, taylor, median_exp


# ---------------------------------------------------------
# 1. UNIDO Manufacturing Output Analysis (Figure 6 / Slide 12)
# ---------------------------------------------------------
def run_manufacturing_analysis(unido, exposure, median_exp):
    df = unido.merge(exposure[["iso3", "region", "exposure", "exposure_pct", "high_exposure"]], on="iso3", how="inner")

    # Time series: Group monthly rolling mean (monthly reporting countries only)
    monthly_countries = df[df["freq"] == "M"].copy()
    group_paths = monthly_countries.groupby(["date", "high_exposure"])["mfg_yoy"].mean().unstack()
    # Filter months with sufficient reporting countries (at least 10)
    obs_counts = monthly_countries.groupby(["date", "high_exposure"])["iso3"].nunique().unstack()
    valid_dates = obs_counts[obs_counts.min(axis=1) >= 10].index
    group_paths = group_paths.loc[valid_dates].rolling(3, min_periods=2).mean()

    # Pre-war vs War windows for DiD: use monthly series if available, else quarterly
    country_rows = []
    for iso, grp in df.groupby("iso3"):
        monthly = grp[grp["freq"] == "M"]
        use = monthly if len(monthly) else grp[grp["freq"] == "Q"]
        pre_b = use[(use["date"] >= "2025-03-01") & (use["date"] <= "2026-02-01")]["mfg_yoy"].mean()
        war_m = use[(use["date"] >= "2026-03-01") & (use["date"] <= "2026-12-01")]["mfg_yoy"].mean()
        pre_a = use[(use["date"] >= "2024-03-01") & (use["date"] <= "2025-02-01")]["mfg_yoy"].mean()
        high_val = grp["high_exposure"].iloc[0]
        country_rows.append({
            "iso3": iso,
            "region": grp["region"].iloc[0],
            "exposure_pct": grp["exposure_pct"].iloc[0],
            "high_exposure": high_val,
            "pre_base": pre_b,
            "war_mean": war_m,
            "pre_alt": pre_a,
            "delta_base": war_m - pre_b,
            "delta_alt": war_m - pre_a,
        })

    country_df = pd.DataFrame(country_rows).dropna(subset=["delta_base"]).copy()

    # DiD OLS regressions with HC1 standard errors
    X_base = sm.add_constant(country_df["high_exposure"])
    mod_base = sm.OLS(country_df["delta_base"], X_base).fit(cov_type="HC1")

    alt_sample = country_df.dropna(subset=["delta_alt"])
    X_alt = sm.add_constant(alt_sample["high_exposure"])
    mod_alt = sm.OLS(alt_sample["delta_alt"], X_alt).fit(cov_type="HC1")

    did_base_coef = mod_base.params["high_exposure"]
    did_base_p = mod_base.pvalues["high_exposure"]
    did_alt_coef = mod_alt.params["high_exposure"]
    did_alt_p = mod_alt.pvalues["high_exposure"]

    # Plot 2-panel figure: Left = Rolling Group Paths, Right = Country Changes vs Exposure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.5), dpi=200)

    # Panel 1: Rolling manufacturing growth
    plot_series = group_paths[group_paths.index >= "2024-01-01"]
    end_date = plot_series.dropna().index.max()
    ax1.axvspan(pd.Timestamp("2026-03-01"), end_date + pd.Timedelta(days=15), color="#f5e6de", alpha=0.5)
    ax1.axhline(0, color=SLATE, lw=0.8, linestyle="--")

    ax1.plot(plot_series.index, plot_series[1], color=ACCENT, lw=2.4, label="High exposure group")
    ax1.plot(plot_series.index, plot_series[0], color=NAVY, lw=2.4, label="Low exposure group")

    ax1.set_ylabel("Manufacturing output, YoY (%)", fontsize=11)
    ax1.set_title("A. Manufacturing Growth (3-month rolling mean)", loc="left", fontsize=12)
    ax1.legend(loc="upper left", frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xlim(pd.Timestamp("2024-01-01"), end_date + pd.Timedelta(days=20))

    # Panel 2: Cross-country change vs Exposure
    ax2.axhline(0, color=SLATE, lw=0.8, linestyle="--")
    ax2.axvline(median_exp * 100, color=ACCENT, lw=1.0, linestyle=":", label="Median split (0.37%)")

    for reg, col in [("Europe", "#2166ac"), ("Asia", "#b2182b")]:
        sub = country_df[country_df["region"] == reg]
        ax2.scatter(sub["exposure_pct"], sub["delta_base"], color=col, s=55, label=reg, edgecolors="white", lw=0.6, zorder=3)

    # Annotate key economies
    key_isos = ["KOR", "THA", "PAK", "IND", "JPN", "DEU", "TUR", "LTU", "GRC", "VNM"]
    for _, r in country_df[country_df["iso3"].isin(key_isos)].iterrows():
        ax2.annotate(r["iso3"], (r["exposure_pct"], r["delta_base"]), xytext=(5, 2), textcoords="offset points",
                     fontsize=8.5, color=INK)

    ax2.set_xlabel("Gulf energy imports (% of GDP)", fontsize=11)
    ax2.set_ylabel("Change in manufacturing growth (pp)", fontsize=11)
    ax2.set_title(f"B. Growth Change by Country (DiD = {did_base_coef:+.2f} pp, p = {did_base_p:.3f})", loc="left", fontsize=12)
    ax2.legend(loc="lower right", frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_manufacturing_iip.png")
    plt.close()

    return {
        "did_base_coef": did_base_coef,
        "did_base_p": did_base_p,
        "did_alt_coef": did_alt_coef,
        "did_alt_p": did_alt_p,
        "n_countries": len(country_df),
        "high_mean": country_df[country_df["high_exposure"] == 1]["delta_base"].mean(),
        "low_mean": country_df[country_df["high_exposure"] == 0]["delta_base"].mean(),
    }


# ---------------------------------------------------------
# 2. Monetary Policy & Taylor Rule (Figure 7 / Slide 14)
# ---------------------------------------------------------
def run_taylor_analysis(taylor):
    # Clean available central banks with actual rate adjustments
    df = taylor.dropna(subset=["d_actual", "d_star"]).sort_values("d_star", ascending=True).copy()

    n_banks = len(df)
    n_lookthrough = int((df["lookthrough"] == 1).sum())
    median_actual = df["d_actual"].median()
    median_prescribed = df["d_star"].median()

    # Specific central bank examples
    ecb = df[df["iso3"] == "EA20"].iloc[0]
    kor = df[df["iso3"] == "KOR"].iloc[0]

    # Plot actual vs Taylor prescriptions
    fig, ax = plt.subplots(figsize=(10.5, 7.8), dpi=200)
    y_pos = np.arange(len(df))

    # Connector lines
    ax.hlines(y_pos, df["d_actual"], df["d_star"], color="#d5dde3", lw=2, zorder=1)

    # Actual rate changes
    ax.scatter(df["d_actual"], y_pos, s=50, color=NAVY, label="Actual policy rate change", zorder=3)
    # Headline Taylor prescribed changes
    ax.scatter(df["d_star"], y_pos, s=50, color=ACCENT, label="Headline Taylor rule prescription", zorder=3)

    # Core Taylor prescribed changes where available
    core_mask = df["d_star_core"].notna()
    ax.scatter(
        df.loc[core_mask, "d_star_core"], y_pos[core_mask],
        s=44, marker="D", facecolors="white", edgecolors=SAGE, linewidths=1.5,
        label="Core Taylor rule prescription", zorder=4,
    )

    ax.axvline(0, color=INK, lw=0.8, linestyle="-")

    labels = [f"{r.iso3} ({r.country})" for r in df.itertuples()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)

    # Highlight ECB
    for tick in ax.get_yticklabels():
        if "EA20" in tick.get_text():
            tick.set_fontweight("bold")
            tick.set_color(NAVY)

    ax.set_xlabel("Change in interest rate since February 2026 (percentage points)", fontsize=11)
    ax.set_title(f"Central Bank Adjustments vs Taylor Rule ({n_lookthrough} of {n_banks} banks looked through)",
                 fontsize=12.5, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_taylor_lookthrough.png")
    plt.close()

    return {
        "n_banks": n_banks,
        "n_lookthrough": n_lookthrough,
        "median_actual": median_actual,
        "median_prescribed": median_prescribed,
        "ecb_actual": ecb["d_actual"],
        "ecb_headline": ecb["d_star"],
        "ecb_core": ecb["d_star_core"],
        "kor_actual": kor["d_actual"],
        "kor_headline": kor["d_star"],
        "kor_core": kor["d_star_core"],
    }


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
def main():
    print("=" * 65)
    print("ECO331 Project: Real Economy & Monetary Policy Analysis")
    print("=" * 65)

    exposure, unido, taylor, median_exp = load_data()

    print("\n1. Manufacturing Output (UNIDO IIP) Analysis:")
    mfg_res = run_manufacturing_analysis(unido, exposure, median_exp)
    print(f"   Sample: {mfg_res['n_countries']} economies")
    print(f"   Baseline DiD (Mar 2025-Feb 2026 pre-war base): {mfg_res['did_base_coef']:+.2f} pp (p = {mfg_res['did_base_p']:.3f})")
    print(f"   High exposure mean change: {mfg_res['high_mean']:+.2f} pp | Low exposure: {mfg_res['low_mean']:+.2f} pp")
    print(f"   Alternative DiD (2024-25 pre-war base): {mfg_res['did_alt_coef']:+.2f} pp (p = {mfg_res['did_alt_p']:.3f})")

    print("\n2. Monetary Policy Benchmark (Taylor Rule):")
    t_res = run_taylor_analysis(taylor)
    print(f"   Central banks analyzed: {t_res['n_banks']}")
    print(f"   Banks moving less than headline Taylor rule: {t_res['n_lookthrough']} of {t_res['n_banks']}")
    print(f"   Median actual rate change: {t_res['median_actual']:+.2f} pp vs prescribed: {t_res['median_prescribed']:+.2f} pp")
    print(f"   ECB  : Actual {t_res['ecb_actual']:+.2f} pp vs Headline {t_res['ecb_headline']:+.2f} pp vs Core {t_res['ecb_core']:+.2f} pp")
    print(f"   Korea: Actual {t_res['kor_actual']:+.2f} pp vs Headline {t_res['kor_headline']:+.2f} pp vs Core {t_res['kor_core']:+.2f} pp")

    print("\n[✓] Output & Taylor rule analysis completed. Figures saved to figures/")
    print("=" * 65)


if __name__ == "__main__":
    main()
