# Hormuz 2026: Energy Supply Shocks, Inflation Pass-Through, and Monetary Policy

**ECO331 Project | IIT Kanpur | November 2026**
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
