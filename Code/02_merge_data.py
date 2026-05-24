"""
02_merge_data.py
Merges Form 5500 plan-level data with state-year covariates for DML analysis.

Produces:
  - Datasets/dml_analysis_data.csv  (state-year level, main DML dataset)
  - Datasets/plan_level_data.parquet (plan-level, for robustness checks)

Run from the Code/ directory: python 02_merge_data.py
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths (relative to the Code/ directory — run this script from there) ─────
# Processed Form 5500 data (~400MB, not included — built from raw Form 5500 filings; see README "Data")
DATA_PATH = '../Datasets/processed_combined_data_1999-2023.xlsx'
COV_DIR = '../Datasets/covariates'
OUT_DIR = '../Datasets'

# ── ACA state definitions ─────────────────────────────────────────────────
ACA_STATES = [
    'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'HI', 'IL',
    'IN', 'IA', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',
    'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'UT', 'VT', 'VA', 'WA'
]

# Years to include (matching original analysis)
YEARS = [y for y in range(2001, 2024) if y != 2004]


def load_form5500():
    """Load and combine all year sheets from the processed Form 5500 data."""
    print("Loading Form 5500 data...")
    excel_file = pd.ExcelFile(DATA_PATH)

    all_data = []
    for year in YEARS:
        sheet_name = f"Year_{year}"
        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # Add year
        df['year'] = year

        # Determine state column
        if year < 2009:
            state_col = 'SPONS_DFE_STATE'
        else:
            state_col = 'SPONS_DFE_MAIL_US_STATE'

        # Standardize state column
        df['state'] = df[state_col].astype(str).str.strip().str.upper()

        # Keep relevant columns
        cols = ['year', 'state', 'ID',
                'WLFR_PREMIUM_RCVD_AMT', 'WLFR_TOT_CHARGES_PAID_AMT',
                'INS_PRSN_COVERED_EOY_CNT']

        # Also grab per-capita if it exists
        if 'WLFR_PREMIUM_RCVD_AMT_PER_CAPITA' in df.columns:
            cols.append('WLFR_PREMIUM_RCVD_AMT_PER_CAPITA')
        if 'WLFR_TOT_CHARGES_PAID_AMT_PER_CAPITA' in df.columns:
            cols.append('WLFR_TOT_CHARGES_PAID_AMT_PER_CAPITA')

        existing_cols = [c for c in cols if c in df.columns]
        all_data.append(df[existing_cols])

    combined = pd.concat(all_data, ignore_index=True)
    print(f"  Loaded {len(combined):,} plan-level observations across {len(YEARS)} years")
    return combined


def clean_plan_data(df):
    """Clean plan-level data: compute per-capita, remove outliers, log-transform."""
    print("Cleaning plan-level data...")

    # Compute per-capita if not already present
    if 'WLFR_PREMIUM_RCVD_AMT_PER_CAPITA' not in df.columns:
        df['WLFR_PREMIUM_RCVD_AMT_PER_CAPITA'] = (
            df['WLFR_PREMIUM_RCVD_AMT'] / df['INS_PRSN_COVERED_EOY_CNT']
        )
    if 'WLFR_TOT_CHARGES_PAID_AMT_PER_CAPITA' not in df.columns:
        df['WLFR_TOT_CHARGES_PAID_AMT_PER_CAPITA'] = (
            df['WLFR_TOT_CHARGES_PAID_AMT'] / df['INS_PRSN_COVERED_EOY_CNT']
        )

    # Replace inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Filter to valid US states only
    valid_states = set(ACA_STATES + ['AL', 'FL', 'GA', 'ID', 'KS', 'TN', 'TX', 'WI', 'WY', 'WV'])
    df = df[df['state'].isin(valid_states)].copy()

    # Create treatment variables
    df['ACA_state'] = df['state'].isin(ACA_STATES).astype(int)
    df['Post_ACA'] = (df['year'] >= 2010).astype(int)
    df['ACA_x_Post'] = df['ACA_state'] * df['Post_ACA']

    # ── Experience-rated plans (premiums) ──
    exp = df[df['WLFR_PREMIUM_RCVD_AMT_PER_CAPITA'] > 0].copy()
    q05 = exp['WLFR_PREMIUM_RCVD_AMT_PER_CAPITA'].quantile(0.05)
    q95 = exp['WLFR_PREMIUM_RCVD_AMT_PER_CAPITA'].quantile(0.95)
    exp = exp[exp['WLFR_PREMIUM_RCVD_AMT_PER_CAPITA'].between(q05, q95)]
    exp['log_premium_per_capita_exp'] = np.log(exp['WLFR_PREMIUM_RCVD_AMT_PER_CAPITA'])
    exp['plan_type'] = 'experience_rated'

    # ── Non-experience-rated plans (charges) ──
    nonexp = df[df['WLFR_TOT_CHARGES_PAID_AMT_PER_CAPITA'] > 0].copy()
    q05c = nonexp['WLFR_TOT_CHARGES_PAID_AMT_PER_CAPITA'].quantile(0.05)
    q95c = nonexp['WLFR_TOT_CHARGES_PAID_AMT_PER_CAPITA'].quantile(0.95)
    nonexp = nonexp[nonexp['WLFR_TOT_CHARGES_PAID_AMT_PER_CAPITA'].between(q05c, q95c)]
    nonexp['log_premium_per_capita_nonexp'] = np.log(nonexp['WLFR_TOT_CHARGES_PAID_AMT_PER_CAPITA'])
    nonexp['plan_type'] = 'non_experience_rated'

    print(f"  Experience-rated plans (after outlier removal): {len(exp):,}")
    print(f"  Non-experience-rated plans (after outlier removal): {len(nonexp):,}")

    return exp, nonexp


def aggregate_to_state_year(exp_df, nonexp_df):
    """Aggregate plan-level data to state-year level."""
    print("Aggregating to state-year level...")

    # Experience-rated aggregation
    exp_agg = exp_df.groupby(['state', 'year']).agg(
        mean_log_premium_exp=('log_premium_per_capita_exp', 'mean'),
        median_log_premium_exp=('log_premium_per_capita_exp', 'median'),
        sd_log_premium_exp=('log_premium_per_capita_exp', 'std'),
        mean_premium_exp=('WLFR_PREMIUM_RCVD_AMT_PER_CAPITA', 'mean'),
        n_plans_exp=('log_premium_per_capita_exp', 'count'),
        mean_persons_covered_exp=('INS_PRSN_COVERED_EOY_CNT', 'mean'),
    ).reset_index()

    # Non-experience-rated aggregation
    nonexp_agg = nonexp_df.groupby(['state', 'year']).agg(
        mean_log_premium_nonexp=('log_premium_per_capita_nonexp', 'mean'),
        median_log_premium_nonexp=('log_premium_per_capita_nonexp', 'median'),
        sd_log_premium_nonexp=('log_premium_per_capita_nonexp', 'std'),
        mean_premium_nonexp=('WLFR_TOT_CHARGES_PAID_AMT_PER_CAPITA', 'mean'),
        n_plans_nonexp=('log_premium_per_capita_nonexp', 'count'),
        mean_persons_covered_nonexp=('INS_PRSN_COVERED_EOY_CNT', 'mean'),
    ).reset_index()

    # Merge experience and non-experience
    merged = pd.merge(exp_agg, nonexp_agg, on=['state', 'year'], how='outer')

    # Add treatment variables
    merged['ACA_state'] = merged['state'].isin(ACA_STATES).astype(int)
    merged['Post_ACA'] = (merged['year'] >= 2010).astype(int)
    merged['ACA_x_Post'] = merged['ACA_state'] * merged['Post_ACA']

    print(f"  State-year observations: {len(merged)}")
    print(f"  States: {merged['state'].nunique()}")
    print(f"  Years: {sorted(merged['year'].unique())}")

    return merged


def load_covariates():
    """Load all covariate files and merge into single state-year panel."""
    print("Loading covariates...")

    covariate_dfs = []

    # Helper to load a CSV, skipping comment lines
    def load_csv(filename):
        filepath = os.path.join(COV_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: {filename} not found, skipping")
            return None
        try:
            df = pd.read_csv(filepath, comment='#')
            if df.empty or df.dropna(how='all').empty:
                print(f"  WARNING: {filename} is empty/all NaN, skipping")
                return None
            return df
        except Exception as e:
            print(f"  ERROR loading {filename}: {e}")
            return None

    # Unemployment
    df = load_csv('unemployment_rate.csv')
    if df is not None and 'unemployment_rate' in df.columns:
        covariate_dfs.append(df[['state', 'year', 'unemployment_rate']])
        print(f"  unemployment_rate: {df.dropna(subset=['unemployment_rate']).shape[0]} obs")

    # Census ACS (income, population)
    df = load_csv('census_acs.csv')
    if df is not None:
        cols = [c for c in ['state', 'year', 'median_household_income', 'population'] if c in df.columns]
        covariate_dfs.append(df[cols])
        print(f"  census_acs: {len(df)} obs")

    # Demographics (age 65+, race)
    df = load_csv('census_demographics.csv')
    if df is not None:
        cols = [c for c in ['state', 'year', 'pct_65plus', 'pct_nh_white'] if c in df.columns]
        covariate_dfs.append(df[cols])
        print(f"  demographics: {len(df)} obs")

    # Poverty and uninsurance
    df = load_csv('poverty_uninsurance.csv')
    if df is not None:
        cols = [c for c in ['state', 'year', 'poverty_rate', 'uninsurance_rate'] if c in df.columns]
        covariate_dfs.append(df[cols])
        print(f"  poverty/uninsurance: {len(df)} obs")

    # Medicaid expansion
    df = load_csv('medicaid_expansion.csv')
    if df is not None:
        covariate_dfs.append(df[['state', 'year', 'medicaid_expansion']])
        print(f"  medicaid_expansion: {len(df)} obs")

    # Marketplace type
    df = load_csv('marketplace_type.csv')
    if df is not None:
        # One-hot encode marketplace type
        df['marketplace_SBM'] = (df['marketplace_type'] == 'SBM').astype(int)
        covariate_dfs.append(df[['state', 'year', 'marketplace_SBM']])
        print(f"  marketplace_type: {len(df)} obs")

    # Pre-ACA guaranteed issue (time-invariant)
    df = load_csv('pre_aca_guaranteed_issue.csv')
    if df is not None:
        covariate_dfs.append(df[['state', 'pre_aca_guaranteed_issue']])
        print(f"  pre_aca_guaranteed_issue: {len(df)} obs")

    # Healthcare spending
    df = load_csv('healthcare_spending_per_capita.csv')
    if df is not None and 'healthcare_spending_per_capita' in df.columns:
        non_null = df.dropna(subset=['healthcare_spending_per_capita'])
        if len(non_null) > 0:
            covariate_dfs.append(df[['state', 'year', 'healthcare_spending_per_capita']])
            print(f"  healthcare_spending: {len(non_null)} obs")

    # Merge all covariates
    if not covariate_dfs:
        print("  WARNING: No covariates loaded!")
        return pd.DataFrame()

    # Start with time-varying covariates (state-year)
    time_varying = [df for df in covariate_dfs if 'year' in df.columns]
    time_invariant = [df for df in covariate_dfs if 'year' not in df.columns]

    if time_varying:
        covariates = time_varying[0]
        for df in time_varying[1:]:
            covariates = pd.merge(covariates, df, on=['state', 'year'], how='outer')
    else:
        covariates = pd.DataFrame()

    for df in time_invariant:
        if not covariates.empty:
            covariates = pd.merge(covariates, df, on='state', how='left')
        else:
            covariates = df

    print(f"  Combined covariates: {covariates.shape}")
    return covariates


def add_year_dummies(df):
    """Add year fixed effects (dummies) to absorb common time trends."""
    # Use 2009 as reference year (last pre-ACA year)
    years_for_dummies = [y for y in df['year'].unique() if y != 2009]
    for y in sorted(years_for_dummies):
        df[f'year_{y}'] = (df['year'] == y).astype(int)
    return df


def print_summary(df):
    """Print summary statistics and diagnostics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")
    print(f"States: {df['state'].nunique()}")
    print(f"Years: {df['year'].min()}-{df['year'].max()}")
    print(f"Treatment states: {df[df['ACA_state']==1]['state'].nunique()}")
    print(f"Control states: {df[df['ACA_state']==0]['state'].nunique()}")

    print(f"\nTreatment states: {sorted(df[df['ACA_state']==1]['state'].unique())}")
    print(f"Control states: {sorted(df[df['ACA_state']==0]['state'].unique())}")

    # Outcome variables
    outcome_cols = [c for c in df.columns if 'log_premium' in c or 'mean_premium' in c]
    if outcome_cols:
        print(f"\nOutcome variables:")
        print(df[outcome_cols].describe().round(4).to_string())

    # Covariate columns
    cov_cols = [c for c in df.columns if c not in
                ['state', 'year', 'ACA_state', 'Post_ACA', 'ACA_x_Post'] +
                outcome_cols +
                [c for c in df.columns if c.startswith('year_')] +
                [c for c in df.columns if 'n_plans' in c or 'sd_' in c or 'persons' in c]]
    if cov_cols:
        print(f"\nCovariate summary:")
        print(df[cov_cols].describe().round(2).to_string())

    # Missingness report
    print(f"\nMissingness report:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
    missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
    missing_df = missing_df[missing_df['missing_count'] > 0]
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("  No missing values!")

    # State-year coverage matrix
    print(f"\nState-year coverage (experience-rated plans count):")
    if 'n_plans_exp' in df.columns:
        pivot = df.pivot_table(values='n_plans_exp', index='state', columns='year',
                               aggfunc='sum', fill_value=0)
        print(f"  Full matrix: {pivot.shape[0]} states × {pivot.shape[1]} years")
        print(f"  Min plans per state-year: {pivot.values[pivot.values > 0].min():.0f}")
        print(f"  Max plans per state-year: {pivot.values.max():.0f}")
        print(f"  Mean plans per state-year: {pivot.values[pivot.values > 0].mean():.0f}")
        zero_cells = (pivot.values == 0).sum()
        if zero_cells > 0:
            print(f"  WARNING: {zero_cells} empty state-year cells")


def main():
    print("=" * 60)
    print("PHASE 1: Building DML Analysis Dataset")
    print("=" * 60)

    # Step 1: Load and clean plan-level data
    raw = load_form5500()
    exp_df, nonexp_df = clean_plan_data(raw)

    # Step 2: Save plan-level data for robustness
    print("\nSaving plan-level data...")
    plan_cols = ['year', 'state', 'ACA_state', 'Post_ACA', 'ACA_x_Post',
                 'WLFR_PREMIUM_RCVD_AMT_PER_CAPITA', 'INS_PRSN_COVERED_EOY_CNT',
                 'log_premium_per_capita_exp']
    plan_existing = [c for c in plan_cols if c in exp_df.columns]
    plan_path = os.path.join(OUT_DIR, 'plan_level_data.parquet')
    exp_df[plan_existing].to_parquet(plan_path, index=False)
    print(f"  Saved {len(exp_df):,} experience-rated plan records to {plan_path}")

    # Step 3: Aggregate to state-year
    state_year = aggregate_to_state_year(exp_df, nonexp_df)

    # Step 4: Load and merge covariates
    covariates = load_covariates()
    if not covariates.empty:
        # Ensure types match
        covariates['year'] = covariates['year'].astype(int)
        state_year['year'] = state_year['year'].astype(int)
        merged = pd.merge(state_year, covariates, on=['state', 'year'], how='left')
    else:
        merged = state_year

    # Step 5: Add year dummies
    merged = add_year_dummies(merged)

    # Step 6: Save
    out_path = os.path.join(OUT_DIR, 'dml_analysis_data.csv')
    merged.to_csv(out_path, index=False)
    print(f"\nSaved analysis dataset to {out_path}")

    # Step 7: Summary
    print_summary(merged)

    # Also save summary stats to Results
    results_dir = '../Results/tables'
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'summary_statistics.csv')
    summary_cols = [c for c in merged.columns if not c.startswith('year_')]
    merged[summary_cols].describe().round(4).to_csv(summary_path)
    print(f"Saved summary statistics to {summary_path}")

    return merged


if __name__ == '__main__':
    df = main()
