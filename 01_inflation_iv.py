"""
Hormuz 2026: Energy Supply Shocks and Inflation Pass-Through
ECO331 Project - IIT Kanpur

This script reproduces the core price pass-through analysis:
1. Exposure calculation and distribution (Asia vs Europe)
2. Shock timeline (Hormuz shipping collapse and Brent surge)
3. Rolling inflation trajectories and parallel pre-trends
4. Difference-in-Differences (DiD) for Headline, Food, Energy, and Core CPI
5. Two-Stage Least Squares (2SLS) IV estimates of oil price pass-through
"""

import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv.model import IV2SLS

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Visual styling
NAVY = "#1f4e79"
ACCENT = "#b85c38"
SAGE = "#5b7c99"
SLATE = "#5a6b7c"
LIGHT_BG = "#f8f9fa"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#d0d7de",
    "axes.linewidth": 0.8,
    "grid.color": "#eaeef2",
    "grid.linewidth": 0.7,
})


def load_data():
    exposure = pd.read_csv(DATA_DIR / "exposure.csv")
    global_daily = pd.read_csv(DATA_DIR / "global_daily.csv", parse_dates=["date"])
    global_monthly = pd.read_csv(DATA_DIR / "global_monthly.csv", parse_dates=["month"])
    panel = pd.read_csv(DATA_DIR / "panel_monthly.csv", parse_dates=["date"])

    # Clean exposure: drop aggregates (EA20) and non-reporters (BGD)
    exposure = exposure.dropna(subset=["exposure"]).copy()
    exposure = exposure[exposure["iso3"] != "EA20"].copy()
    exposure["exposure_pct"] = exposure["exposure"] * 100
    median_exp = exposure["exposure"].median()
    exposure["high_exposure"] = (exposure["exposure"] >= median_exp).astype(int)

    # Calculate cross-country standard deviation of exposure (1.36% of GDP)
    sd_scale = exposure["exposure"].std()
    exposure["exposure_sd"] = exposure["exposure"] / sd_scale

    return exposure, global_daily, global_monthly, panel, median_exp, sd_scale


# ---------------------------------------------------------
# 1. Shock Timeline (Figure 1 / Slide 3)
# ---------------------------------------------------------
def plot_shock_timeline(daily):
    df = daily[daily["date"] >= "2025-01-01"].sort_values("date").copy()
    transits = df.dropna(subset=["n_total"]).set_index("date")["n_total"]
    rolling_transits = transits.rolling("7D", min_periods=4).mean()

    brent = df.dropna(subset=["brent"])

    fig, ax1 = plt.subplots(figsize=(10, 5.2), dpi=200)

    # Shade war window (March 2026 onwards)
    ax1.axvspan(pd.Timestamp("2026-03-01"), df["date"].max(), color="#f5e6de", alpha=0.5, label="War period")

    # Key timeline milestones
    ax1.axvline(pd.Timestamp("2026-02-28"), color=ACCENT, linestyle="--", linewidth=1.2)
    ax1.text(pd.Timestamp("2026-03-03"), 105, "Closure\n(28 Feb)", fontsize=9, color=ACCENT)
    ax1.axvline(pd.Timestamp("2026-06-18"), color=SAGE, linestyle=":", linewidth=1.2)
    ax1.text(pd.Timestamp("2026-06-20"), 105, "Memo\n(18 Jun)", fontsize=9, color=SAGE)
    ax1.axvline(pd.Timestamp("2026-07-06"), color=ACCENT, linestyle="--", linewidth=1.2)
    ax1.text(pd.Timestamp("2026-07-08"), 105, "Re-closure\n(6 Jul)", fontsize=9, color=ACCENT)

    # Transits line
    line1 = ax1.plot(rolling_transits.index, rolling_transits, color=ACCENT, lw=2.2, label="Hormuz transits (7-day mean)")
    ax1.set_ylabel("Hormuz transits (ships / day)", color=ACCENT, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=ACCENT)
    ax1.set_ylim(0, 130)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Brent spot price on secondary axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(brent["date"], brent["brent"], color=NAVY, lw=1.8, label="Brent crude spot price")
    ax2.set_ylabel("Brent crude (USD / barrel)", color=NAVY, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=NAVY)
    ax2.set_ylim(40, 140)
    ax2.grid(False)

    ax1.set_xlim(pd.Timestamp("2025-01-01"), df["date"].max() + pd.Timedelta(days=5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.title("Hormuz Tanker Transits and Brent Crude Oil Price (2025–2026)", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_shock_timeline.png")
    plt.close()


# ---------------------------------------------------------
# 2. Exposure Distribution (Figure 2 / Slide 8)
# ---------------------------------------------------------
def plot_exposure(exposure, median_exp):
    sorted_df = exposure.sort_values("exposure_pct", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(7.5, 9.2), dpi=200)
    y_pos = np.arange(len(sorted_df))
    colors = [NAVY if reg == "Europe" else SAGE for reg in sorted_df["region"]]

    bars = ax.barh(y_pos, sorted_df["exposure_pct"], color=colors, height=0.72)

    # Hatch Singapore (re-export hub)
    for bar, hub in zip(bars, sorted_df["hub_flag"]):
        if hub == 1:
            bar.set_hatch("////")
            bar.set_edgecolor(ACCENT)
            bar.set_linewidth(0.8)

    # Median line
    median_pct = median_exp * 100
    n_low = (sorted_df["exposure_pct"] < median_pct).sum()
    ax.axhline(n_low - 0.5, color=ACCENT, linestyle="--", linewidth=1.2)
    ax.text(sorted_df["exposure_pct"].max() * 0.95, n_low - 0.5 + 0.3,
            f"Median = {median_pct:.2f}% of GDP", ha="right", va="bottom",
            color=ACCENT, fontweight="bold", fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df["country"], fontsize=9.5)
    ax.set_xlabel("Gulf Energy Imports (% of GDP, 2023–25 average)", fontsize=11)
    ax.set_xlim(0, sorted_df["exposure_pct"].max() * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for i, val in enumerate(sorted_df["exposure_pct"]):
        ax.text(val + 0.05, i, f"{val:.2f}", va="center", fontsize=8.5, color=SLATE)

    legend_elements = [
        Patch(facecolor=NAVY, label="Europe (median 0.13%)"),
        Patch(facecolor=SAGE, label="Asia (median 2.18%)"),
        Patch(facecolor=SAGE, hatch="////", edgecolor=ACCENT, label="Singapore (re-export hub)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white")
    plt.title("Pre-War Exposure to Persian Gulf Energy Imports", fontsize=12.5, pad=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_exposure_bar.png")
    plt.close()


# ---------------------------------------------------------
# 3. Inflation Trajectories (Figure 3 / Slide 9)
# ---------------------------------------------------------
def plot_inflation_paths(panel, exposure):
    df = panel.merge(exposure[["iso3", "high_exposure"]], on="iso3", how="inner")
    df = df[df["date"] >= "2025-01-01"].copy()

    # 3-month rolling average headline inflation
    group_means = df.groupby(["date", "high_exposure"])["cpi_headline_yoy"].mean().unstack()
    group_means = group_means.rolling(3, min_periods=2).mean()

    # Changes relative to February 2026
    feb_base = group_means.loc["2026-02-01"]
    delta_headline = group_means - feb_base

    # Regional levels (excluding Turkey outlier)
    euro_clean = df[(df["region"] == "Europe") & (df["iso3"] != "TUR")].groupby("date")["cpi_headline_yoy"].mean()
    asia_levels = df[df["region"] == "Asia"].groupby("date")["cpi_headline_yoy"].mean()

    euro_roll = euro_clean.rolling(3, min_periods=2).mean()
    asia_roll = asia_levels.rolling(3, min_periods=2).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.2), dpi=200, sharex=True)

    # Panel A: Change since Feb 2026
    ax1.axvspan(pd.Timestamp("2026-03-01"), delta_headline.index.max(), color="#f5e6de", alpha=0.5)
    ax1.axhline(0, color="#9aa5b1", lw=0.8)
    ax1.plot(delta_headline.index, delta_headline[1], color=ACCENT, lw=2.4, label="High exposure (20 economies)")
    ax1.plot(delta_headline.index, delta_headline[0], color=NAVY, lw=2.4, label="Low exposure (20 economies)")

    last_date = delta_headline.dropna().index.max()
    d_high = delta_headline.loc[last_date, 1]
    d_low = delta_headline.loc[last_date, 0]
    ax1.annotate(f"+{d_high:.1f} pp", (last_date, d_high), xytext=(8, 0), textcoords="offset points",
                 fontweight="bold", color=ACCENT)
    ax1.annotate(f"+{d_low:.1f} pp", (last_date, d_low), xytext=(8, 0), textcoords="offset points",
                 fontweight="bold", color=NAVY)

    ax1.set_ylabel("Change since Feb 2026 (pp)", fontsize=11)
    ax1.set_title("A. Headline Inflation Change Relative to February 2026", loc="left", fontsize=12)
    ax1.legend(loc="upper left", frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Regional Inflation Levels
    ax2.axvspan(pd.Timestamp("2026-03-01"), euro_roll.index.max(), color="#f5e6de", alpha=0.5)
    ax2.plot(asia_roll.index, asia_roll, color="#b2182b", lw=2.4, label="Asia (12 economies)")
    ax2.plot(euro_roll.index, euro_roll, color="#2166ac", lw=2.4, label="Europe, excluding Turkey (28 economies)")

    ax2.set_ylabel("Headline CPI, YoY (%)", fontsize=11)
    ax2.set_title("B. Headline Inflation Levels by Region", loc="left", fontsize=12)
    ax2.legend(loc="upper left", frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_inflation_stacked.png")
    plt.close()


# ---------------------------------------------------------
# 4. Difference-in-Differences (Figure 4 / Slide 10)
# ---------------------------------------------------------
def run_did_analysis(panel, exposure):
    df = panel.merge(exposure[["iso3", "high_exposure"]], on="iso3", how="inner")

    pre_mask = (df["date"] >= "2025-03-01") & (df["date"] <= "2026-02-01")
    war_mask = (df["date"] >= "2026-03-01") & (df["date"] <= "2026-10-01")

    outcomes = [
        ("cpi_headline_yoy", "Headline"),
        ("cpi_food_yoy", "Food"),
        ("cpi_core_yoy", "Core"),
        ("cpi_energy_yoy", "Energy"),
    ]

    did_results = []
    cell_means = {}

    for var, label in outcomes:
        sub = df[pre_mask | war_mask].dropna(subset=[var]).copy()
        sub["post"] = (sub["date"] >= "2026-03-01").astype(int)

        # Average by country and period
        country_means = sub.groupby(["iso3", "high_exposure", "post"])[var].mean().reset_index()

        # Balance check: keep countries present in both pre and war periods
        counts = country_means.groupby("iso3")["post"].nunique()
        balanced_countries = counts[counts == 2].index
        balanced = country_means[country_means["iso3"].isin(balanced_countries)].copy()

        # Group means for visualization
        g_means = balanced.groupby(["high_exposure", "post"])[var].mean()
        n_countries = balanced.groupby("high_exposure")["iso3"].nunique()

        cell_means[var] = {
            "low_pre": g_means.get((0, 0), np.nan),
            "low_war": g_means.get((0, 1), np.nan),
            "high_pre": g_means.get((1, 0), np.nan),
            "high_war": g_means.get((1, 1), np.nan),
            "n_low": n_countries.get(0, 0),
            "n_high": n_countries.get(1, 0),
        }

        # Saturated OLS DiD: Y = b0 + b1*High + b2*Post + b3*(High*Post) + e
        balanced["post_x_high"] = balanced["post"] * balanced["high_exposure"]
        X = sm.add_constant(balanced[["high_exposure", "post", "post_x_high"]])
        y = balanced[var]

        # Clustered SEs by country
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": balanced["iso3"]})

        did_results.append({
            "outcome": var,
            "label": label,
            "did_coef": model.params["post_x_high"],
            "se": model.bse["post_x_high"],
            "pvalue": model.pvalues["post_x_high"],
            "n_countries": len(balanced_countries),
        })

    # Plot DiD Panels (Headline, Food, Core)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=200, sharey=True)
    width = 0.35

    plot_vars = [("cpi_headline_yoy", "Headline"), ("cpi_food_yoy", "Food"), ("cpi_core_yoy", "Core")]

    for ax, (var, label) in zip(axes, plot_vars):
        cm = cell_means[var]
        res = next(r for r in did_results if r["outcome"] == var)

        # Low exposure bars
        ax.bar(0 - width / 2, cm["low_pre"], width, color="#c9d5e2", zorder=2)
        ax.bar(0 + width / 2, cm["low_war"], width, color=NAVY, zorder=2)
        ax.text(0, max(cm["low_pre"], cm["low_war"]) + 0.25,
                f"{cm['low_war'] - cm['low_pre']:+.2f}", ha="center", color=NAVY, fontweight="bold")
        ax.text(0, -0.4, f"Low exposure\n({cm['n_low']} countries)", ha="center", va="top", fontsize=9.5)

        # High exposure bars
        ax.bar(1 - width / 2, cm["high_pre"], width, color="#f1d3c1", zorder=2)
        ax.bar(1 + width / 2, cm["high_war"], width, color=ACCENT, zorder=2)
        ax.text(1, max(cm["high_pre"], cm["high_war"]) + 0.25,
                f"{cm['high_war'] - cm['high_pre']:+.2f}", ha="center", color=ACCENT, fontweight="bold")
        ax.text(1, -0.4, f"High exposure\n({cm['n_high']} countries)", ha="center", va="top", fontsize=9.5)

        sig = "***" if res["pvalue"] < 0.01 else "**" if res["pvalue"] < 0.05 else "*" if res["pvalue"] < 0.1 else "ns"
        ax.set_title(f"{label}: DiD = {res['did_coef']:+.2f} pp ({sig})", fontsize=12, pad=10)
        ax.set_xticks([])
        ax.set_xlim(-0.7, 1.7)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Average inflation, YoY (%)", fontsize=11)
    axes[0].set_ylim(0, 6.8)

    legend_elements = [
        Patch(facecolor="#d9d9d9", label="Pre-war year (Mar 2025–Feb 2026)"),
        Patch(facecolor="#5a5a5a", label="War months (Mar–Oct 2026)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 1.02), ncol=2, frameon=True)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_did_panels.png")
    plt.close()

    return pd.DataFrame(did_results)


# ---------------------------------------------------------
# 5. Two-Stage Least Squares (2SLS) (Figure 5 / Slide 11)
# ---------------------------------------------------------
def run_2sls_analysis(panel, exposure, global_monthly):
    # Compute 12-month change in Brent and Hormuz transit shortfall relative to 2025 mean
    gm = global_monthly.sort_values("month").copy()
    gm["log_brent"] = np.log(gm["brent"])
    gm["d12_log_brent"] = gm["log_brent"] - gm["log_brent"].shift(12)
    base_2025 = gm.loc[gm["month"].between("2025-01-01", "2025-12-31"), "hormuz_tanker_transits"].mean()
    gm["shortfall"] = 1.0 - gm["hormuz_tanker_transits"] / base_2025
    gm = gm[["month", "d12_log_brent", "shortfall"]].rename(columns={"month": "date"})

    # Prepare war window (Jan 2025 to Oct 2026)
    df = panel.merge(exposure[["iso3", "exposure_sd"]], on="iso3", how="inner")
    df = df.merge(gm, on="date", how="inner")
    df = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2026-10-01")].copy()

    # Endogenous term: 12-month change in log Brent * Exposure_sd
    df["d12_x"] = df["d12_log_brent"] * df["exposure_sd"]
    # Instrument: Hormuz shipping shortfall * Exposure_sd
    df["short_x"] = df["shortfall"] * df["exposure_sd"]

    outcomes = [
        ("cpi_headline_yoy", "Headline"),
        ("cpi_food_yoy", "Food"),
        ("cpi_energy_yoy", "Energy"),
    ]

    results = []

    for var, label in outcomes:
        sub = df.dropna(subset=[var, "d12_x", "short_x"]).copy()

        # Two-way fixed effects demeaning via alternating projections
        cols = [var, "d12_x", "short_x"]
        dm = sub[cols].to_numpy(dtype=float).copy()
        entities = sub["iso3"].to_numpy()
        times = sub["date"].to_numpy()

        for _ in range(15):
            for val in np.unique(entities):
                mask = (entities == val)
                dm[mask] -= dm[mask].mean(axis=0)
            for val in np.unique(times):
                mask = (times == val)
                dm[mask] -= dm[mask].mean(axis=0)

        y = pd.Series(dm[:, 0], name=var)
        x = pd.DataFrame(dm[:, 1], columns=["d12_x"])
        z = pd.DataFrame(dm[:, 2], columns=["short_x"])

        # OLS and 2SLS with country-clustered SEs
        ols = IV2SLS(y, x, None, None).fit(cov_type="clustered", clusters=sub["iso3"])
        tsls = IV2SLS(y, None, x, z).fit(cov_type="clustered", clusters=sub["iso3"])

        # Degrees-of-freedom adjustment for two-way fixed effects
        n_countries = sub["iso3"].nunique()
        n_months = sub["date"].nunique()
        df_absorbed = n_countries + n_months - 1
        df_adj = np.sqrt((len(sub) - 1) / (len(sub) - 1 - df_absorbed))

        ols_se = float(ols.std_errors.iloc[0]) * df_adj
        tsls_se = float(tsls.std_errors.iloc[0]) * df_adj
        ols_coef = float(ols.params.iloc[0])
        tsls_coef = float(tsls.params.iloc[0])

        results.append({
            "outcome": var,
            "label": label,
            "ols_coef": ols_coef,
            "ols_se": ols_se,
            "ols_ci": [ols_coef - 1.96 * ols_se, ols_coef + 1.96 * ols_se],
            "tsls_coef": tsls_coef,
            "tsls_se": tsls_se,
            "tsls_ci": [tsls_coef - 1.96 * tsls_se, tsls_coef + 1.96 * tsls_se],
            "nobs": tsls.nobs,
        })

    # Plot OLS vs 2SLS point estimates & 95% CIs
    fig, ax = plt.subplots(figsize=(7.2, 5.5), dpi=200)
    ax.axvline(0, color="#c5cdd4", linestyle="--", linewidth=1.0)

    y_positions = np.arange(len(outcomes))[::-1]

    for i, res in enumerate(results):
        y = y_positions[i]
        # OLS
        ax.errorbar(
            res["ols_coef"], y + 0.12,
            xerr=[[res["ols_coef"] - res["ols_ci"][0]], [res["ols_ci"][1] - res["ols_coef"]]],
            fmt="o", color=NAVY, capsize=4, elinewidth=1.6, label="OLS" if i == 0 else None,
        )
        # 2SLS
        ax.errorbar(
            res["tsls_coef"], y - 0.12,
            xerr=[[res["tsls_coef"] - res["tsls_ci"][0]], [res["tsls_ci"][1] - res["tsls_coef"]]],
            fmt="s", color=ACCENT, capsize=4, elinewidth=1.6, label="2SLS (IV)" if i == 0 else None,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in results], fontsize=11)
    ax.set_xlabel("Pass-through coefficient (pp per SD of exposure, per log point of Brent)", fontsize=10)
    ax.set_ylim(-0.6, len(outcomes) - 0.4)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, facecolor="white")
    plt.title("Supply-Driven Oil Price Pass-Through: OLS vs 2SLS", fontsize=12.5, pad=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_iv_estimates.png")
    plt.close()

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Main execution and summary reporting
# ---------------------------------------------------------
def main():
    print("=" * 65)
    print("ECO331 Project: Hormuz Shock Inflation Pass-Through Analysis")
    print("=" * 65)

    exposure, global_daily, global_monthly, panel, median_exp, sd_scale = load_data()
    print(f"\n1. Exposure Distribution:")
    print(f"   Countries analyzed: {len(exposure)}")
    print(f"   Sample median exposure: {median_exp*100:.2f}% of GDP")
    print(f"   Exposure 1 SD scale: {sd_scale*100:.2f}% of GDP")
    print(f"   Asia median: {exposure[exposure['region']=='Asia']['exposure_pct'].median():.2f}% | "
          f"Europe median: {exposure[exposure['region']=='Europe']['exposure_pct'].median():.2f}%")

    print("\n2. Generating Figures 1, 2, and 3...")
    plot_shock_timeline(global_daily)
    plot_exposure(exposure, median_exp)
    plot_inflation_paths(panel, exposure)

    print("\n3. Difference-in-Differences (DiD) Estimation:")
    did_df = run_did_analysis(panel, exposure)
    for _, r in did_df.iterrows():
        print(f"   {r['label']:<10}: DiD = {r['did_coef']:+.2f} pp (SE {r['se']:.2f}, p = {r['pvalue']:.3f}) [N={r['n_countries']}]")

    print("\n4. 2SLS Instrumental Variables (IV) Estimation:")
    iv_df = run_2sls_analysis(panel, exposure, global_monthly)
    for _, r in iv_df.iterrows():
        print(f"   {r['label']:<10}: OLS = {r['ols_coef']:.2f} (SE {r['ols_se']:.2f}) | 2SLS = {r['tsls_coef']:.2f} (SE {r['tsls_se']:.2f})")

    print("\n[✓] Inflation & IV analysis completed. Figures saved to figures/")
    print("=" * 65)


if __name__ == "__main__":
    main()
