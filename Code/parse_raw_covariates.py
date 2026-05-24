"""Parse BLS LAUS and CMS health expenditure raw data into clean CSVs.

Run from the Code/ directory: python parse_raw_covariates.py
"""
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "../Datasets/covariates"
BLS_DIR = os.path.join(OUTPUT_DIR, "bls_lau")
CMS_FILE = os.path.join(OUTPUT_DIR, "cms_state_health_expenditure", "Residence_all_tables.xlsx")

STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
    'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
    'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
    'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
    'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
    'VT': '50', 'VA': '51', 'WA': '53', 'WI': '55', 'WY': '56'
}

STATE_NAMES = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# ── BLS LAUS ──
print("Parsing BLS LAUS data...")

state_series = {}
for abbr, fips in STATE_FIPS.items():
    series_id = f"LAUST{fips}0000000000003"
    state_series[series_id] = abbr

all_unemp = []
for fname in sorted(os.listdir(BLS_DIR)):
    if fname.startswith("la.data.") and fname.endswith(".txt"):
        fpath = os.path.join(BLS_DIR, fname)
        print(f"  Reading {fname}...")
        df = pd.read_csv(fpath, sep='\t', dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df['series_id'] = df['series_id'].str.strip()
        df['period'] = df['period'].str.strip()
        df['value'] = df['value'].str.strip()

        mask = df['series_id'].isin(state_series) & (df['period'] == 'M13')
        matched = df[mask]

        for _, row in matched.iterrows():
            state = state_series.get(row['series_id'])
            if state:
                try:
                    all_unemp.append({
                        'state': state,
                        'year': int(row['year']),
                        'unemployment_rate': float(row['value'])
                    })
                except (ValueError, TypeError):
                    pass

unemp_df = pd.DataFrame(all_unemp)
unemp_df = unemp_df[(unemp_df['year'] >= 2001) & (unemp_df['year'] <= 2023)]
unemp_df.to_csv(os.path.join(OUTPUT_DIR, 'unemployment_rate.csv'), index=False)
print(f"  Saved {len(unemp_df)} unemployment records")
print(f"  States: {unemp_df['state'].nunique()}, Years: {unemp_df['year'].min()}-{unemp_df['year'].max()}")

# ── CMS Health Expenditure ──
print("\nParsing CMS health expenditure data...")

cms = pd.read_excel(CMS_FILE, sheet_name='Table 1 Personal Health Care', header=None)

years_row = cms.iloc[1].tolist()
years = []
for v in years_row[1:]:
    try:
        years.append(int(v))
    except (ValueError, TypeError):
        years.append(None)

all_spending = []
for idx in range(2, len(cms)):
    state_name = str(cms.iloc[idx, 0]).strip()
    state_abbr = STATE_NAMES.get(state_name)
    if not state_abbr:
        continue

    for col_idx, year in enumerate(years):
        if year is None or year < 2001 or year > 2023:
            continue
        val = cms.iloc[idx, col_idx + 1]
        try:
            spending_millions = float(val)
            all_spending.append({
                'state': state_abbr,
                'year': year,
                'total_health_spending_millions': spending_millions
            })
        except (ValueError, TypeError):
            pass

spending_df = pd.DataFrame(all_spending)

# Load population to compute per-capita
pop_path = os.path.join(OUTPUT_DIR, 'census_acs.csv')
pop_df = pd.read_csv(pop_path)
if 'population' in pop_df.columns:
    spending_df = spending_df.merge(pop_df[['state', 'year', 'population']], on=['state', 'year'], how='left')
    spending_df['healthcare_spending_per_capita'] = (
        spending_df['total_health_spending_millions'] * 1e6 / spending_df['population']
    )

spending_df[['state', 'year', 'healthcare_spending_per_capita', 'total_health_spending_millions']].to_csv(
    os.path.join(OUTPUT_DIR, 'healthcare_spending_per_capita.csv'), index=False
)
valid = spending_df.dropna(subset=['healthcare_spending_per_capita'])
print(f"  Saved {len(spending_df)} healthcare spending records")
print(f"  Per-capita computed for {len(valid)} obs")
print(f"  States: {spending_df['state'].nunique()}, Years: {spending_df['year'].min()}-{spending_df['year'].max()}")

print("\nDone!")
