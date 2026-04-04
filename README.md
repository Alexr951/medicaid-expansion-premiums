# Medicaid Expansion and Employer-Sponsored Insurance Premiums

Originally an Econ 490 project with Ameesh Kumar. Estimates the causal effect of state Medicaid expansion on employer-sponsored health insurance premiums using Form 5500 filings (2001-2024).

## Data

- **Form 5500 / Schedule A**: Plan-level filings from the Department of Labor. Raw files are large (~1GB) and not included in the repo — download from [EFAST2](https://www.dol.gov/agencies/ebsa/researchers/data/form-5500).
- **Covariates**: Census ACS demographics, BLS unemployment, CMS health spending, marketplace type, and Medicaid expansion timing. Processed versions are in `Datasets/covariates/`.
- **Analysis dataset**: `Datasets/dml_analysis_data.csv` — state-year panel (51 states, 2001-2024, excluding 2004).

## Code

Pipeline runs in order:

1. `Code/01_download_covariates.py` — pulls covariate data from Census, BLS, CMS
2. `Code/02_merge_data.py` — merges Form 5500 with covariates, computes state-year aggregates
3. `Code/parse_raw_covariates.py` — helper for processing raw BLS files
4. `Code/DML_Analysis_revised.ipynb` — main analysis notebook (TWFE, event study, DML, robustness checks)

## Methods

- Two-way fixed effects DiD as baseline
- Event study with staggered adoption (8 expansion cohorts, 2014-2023)
- Double/debiased machine learning (Chernozhukov et al. 2018) with RF, GBR, and Lasso learners
- Primary DML specification includes state fixed effects for within-state identification

## Results

Figures and tables are saved to `Results/`. The main comparison table is in `Results/tables/twfe_vs_dml_comparison.csv`.
