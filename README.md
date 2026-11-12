# Hormuz 2026: Energy Supply Shocks, Inflation Pass-Through, and Monetary Policy

**ECO331 Project | IIT Kanpur | November 2026**

This repository contains the empirical replication pipeline for our study of the macroeconomic transmission of the 2026 Strait of Hormuz closure. 

We examine how the physical disruption of Gulf shipping and subsequent crude oil price surge passed through into consumer prices (headline, food, core, and energy), manufacturing production, and central bank policy rates across 40 economies.

---

## Empirical Strategy & Key Findings

1. **Pre-War Exposure Margin**: We construct pre-war dependence on Persian Gulf energy imports as a share of GDP (2023–2025 UN Comtrade imports divided by World Bank GDP). Exposure is substantially higher in Asia (median: 2.18% of GDP) than in Europe (median: 0.13%). The sample median (0.37%) defines the high- and low-exposure comparison groups.
2. **Difference-in-Differences (DiD)**: Comparing average inflation in the pre-war year (March 2025–February 2026) with the war period (March–October 2026):
   - **Headline CPI**: +1.87 pp in high-exposure economies vs. −0.04 pp in low-exposure economies (DiD: **+1.91 pp**, $p = 0.001$).
   - **Food CPI**: DiD of **+1.86 pp** ($p = 0.033$), reflecting higher freight and fertilizer input costs.
   - **Core CPI**: DiD of **+0.80 pp** ($p = 0.034$), indicating second-round pass-through into non-energy goods and services.
   - **Energy CPI**: Little differential pass-through (+0.84 pp, $p = 0.731$) as Asian governments capped retail fuel prices and absorbed costs fiscally.
3. **2SLS Instrumental Variables**: To isolate supply-driven oil price movements from world demand fluctuations, we instrument the interaction $\Delta_{12}\ln \text{Brent}_t \times \text{Exposure}_i$ with satellite AIS tanker transit shortfalls from IMF PortWatch ($\text{Shortfall}_t \times \text{Exposure}_i$), with two-way country and month fixed effects:
   - 2SLS headline pass-through is **1.75** (SE 0.66) compared to OLS **1.42** (SE 0.55).
   - Food pass-through is highest at **2.52** (SE 1.03), while energy CPI is statistically insignificant (**−2.05**).
4. **Real Output**: Using UNIDO's Index of Industrial Production (IIP), manufacturing output had not contracted by August/September 2026. Growth in exposed economies remained near 4–5%, supported by fuel price subsidies and a global semiconductor export boom.
5. **Monetary Policy Benchmark**: Evaluating central bank policy rates against the textbook Taylor rule ($\Delta i^* = 1.5 \Delta \pi + 0.5 \Delta \tilde{y}$):
   - **14 of 19** central banks adjusted rates by less than the headline Taylor rule (median actual change: 0.0 pp vs. prescribed +2.53 pp).
   - Central banks broadly looked through the imported supply shock, maintaining interest rate smoothing and tracking core inflation benchmarks more closely.

---

## Repository Structure

```
├── README.md                  # Project overview and replication guide
├── requirements.txt           # Python dependencies
├── .gitignore
├── 01_inflation_iv.py         # Exposure, DiD, and 2SLS IV regressions + plots
├── 02_real_and_policy.py      # UNIDO manufacturing response & Taylor rule analysis + plots
├── data/
│   ├── exposure.csv           # Country-level pre-war Gulf exposure (% of GDP)
│   ├── global_daily.csv       # Daily IMF PortWatch tanker transits & Brent spot prices
│   ├── global_monthly.csv     # Monthly Brent, Hormuz transits, and transit shortfalls
│   ├── panel_monthly.csv      # Country-month macro panel (CPI components, policy rates, IP)
│   ├── taylor_by_bank.csv     # Central bank rates, Taylor prescriptions, and look-through tags
│   └── unido_iip_monthly.csv  # UNIDO manufacturing industrial production indices
└── figures/                   # Generated high-resolution figures for presentation slides
    ├── 01_shock_timeline.png
    ├── 02_exposure_bar.png
    ├── 03_inflation_stacked.png
    ├── 04_did_panels.png
    ├── 05_iv_estimates.png
    ├── 06_manufacturing_iip.png
    └── 07_taylor_lookthrough.png
```

---

## Replication Instructions

### 1. Requirements

Install standard scientific Python libraries:
```bash
pip install -r requirements.txt
```

### 2. Run Analysis

Both scripts run self-contained from the repository root:

- **Inflation & IV Pipeline** (Figures 1–5):
  ```bash
  python 01_inflation_iv.py
  ```
  Estimates exposure statistics, difference-in-differences regressions, and two-stage least squares instrumental variable regressions with country-clustered standard errors.

- **Real Economy & Monetary Policy Pipeline** (Figures 6–7):
  ```bash
  python 02_real_and_policy.py
  ```
  Estimates manufacturing growth changes (UNIDO IIP) and counterfactual Taylor rule policy rate paths across 19 central banks.

---

## Data Sources

All datasets are publicly available:
- **Hormuz Daily Tanker Transits**: IMF PortWatch (Satellite AIS ship tracking)
- **Crude Oil Prices**: FRED (St. Louis Fed) & World Bank Pink Sheet
- **Consumer Price Indices**: Eurostat HICP, IMF CPI, and OECD
- **Manufacturing Output**: UNIDO Index of Industrial Production (IIP, ISIC C)
- **Central Bank Policy Rates**: Bank for International Settlements (BIS)
- **Bilateral Trade & GDP**: UN Comtrade (HS 2709, 2710, 2711) & World Bank
