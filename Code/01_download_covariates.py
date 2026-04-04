"""
01_download_covariates.py
Downloads/constructs state-year level covariates for DML analysis.

Sources:
- BLS LAUS: Unemployment rate by state-year
- Census ACS: Median household income, poverty rate, uninsurance rate,
              population, % 65+, % non-Hispanic white
- KFF / hardcoded: Medicaid expansion status, marketplace type
- CMS: Per-capita healthcare spending (placeholder)

Run: python Code/01_download_covariates.py
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Datasets', 'covariates')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# State FIPS mapping
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
FIPS_TO_STATE = {v: k for k, v in STATE_FIPS.items()}

ALL_STATES = sorted(STATE_FIPS.keys())
YEARS = list(range(2001, 2024))


def download_bls_unemployment():
    """Download state-level unemployment rates from BLS LAUS (API v2, no key needed)."""
    print("Downloading BLS unemployment data...")

    # BLS LAUS series IDs: LASST{FIPS}0000000000003 = unemployment rate
    all_data = []

    # BLS API allows 50 series per request, 10 years max
    for year_start in range(2001, 2024, 10):
        year_end = min(year_start + 9, 2023)
        series_ids = [f"LASST{fips}0000000000003" for fips in STATE_FIPS.values()]

        # Split into batches of 50
        for i in range(0, len(series_ids), 50):
            batch = series_ids[i:i+50]
            payload = {
                "seriesid": batch,
                "startyear": str(year_start),
                "endyear": str(year_end),
                "registrationkey": "",  # No key needed for v1
            }

            try:
                # Use v1 (no key required, lower rate limits)
                resp = requests.post(
                    "https://api.bls.gov/publicAPI/v1/timeseries/data/",
                    json=payload,
                    timeout=30
                )
                data = resp.json()

                if data.get('status') == 'REQUEST_SUCCEEDED':
                    for series in data['Results']['series']:
                        series_id = series['seriesID']
                        fips = series_id[5:7]
                        state = FIPS_TO_STATE.get(fips, '')
                        if not state:
                            continue

                        for item in series['data']:
                            if item['period'] == 'M13':  # Annual average
                                all_data.append({
                                    'state': state,
                                    'year': int(item['year']),
                                    'unemployment_rate': float(item['value'])
                                })
                else:
                    print(f"  BLS API warning: {data.get('message', ['Unknown'])[0][:80]}")

                time.sleep(1)  # Rate limit
            except Exception as e:
                print(f"  BLS API error for batch starting {year_start}: {e}")
                time.sleep(2)

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(os.path.join(OUTPUT_DIR, 'unemployment_rate.csv'), index=False)
        print(f"  Saved {len(df)} unemployment records")
        return df
    else:
        print("  BLS download failed — creating placeholder")
        return create_placeholder('unemployment_rate', 'unemployment_rate',
                                  'BLS LAUS: https://www.bls.gov/lau/tables.htm')


def download_census_acs():
    """
    Download Census ACS 1-year estimates for multiple variables.
    Uses the Census API (no key required for small requests, but rate limited).
    """
    print("Downloading Census ACS data...")

    # ACS 1-year variables
    # B19013_001E = median household income
    # S1701_C03_001E = poverty rate (%) -- only in subject tables
    # B01003_001E = total population
    # We'll use the detailed tables API which is more reliable without a key

    variables = {
        'B19013_001E': 'median_household_income',
        'B01003_001E': 'population',
    }

    all_data = []

    # ACS 1-year available from 2005+ (some from 2005, more reliable from 2006+)
    for year in range(2005, 2024):
        var_string = ','.join(variables.keys())
        url = (
            f"https://api.census.gov/data/{year}/acs/acs1"
            f"?get=NAME,{var_string}&for=state:*"
        )

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                headers = data[0]
                for row in data[1:]:
                    row_dict = dict(zip(headers, row))
                    fips = row_dict.get('state', '')
                    state = FIPS_TO_STATE.get(fips, '')
                    if not state:
                        continue

                    record = {'state': state, 'year': year}
                    for census_var, our_name in variables.items():
                        val = row_dict.get(census_var)
                        try:
                            record[our_name] = float(val) if val and val != 'null' else np.nan
                        except (ValueError, TypeError):
                            record[our_name] = np.nan
                    all_data.append(record)

                print(f"  ACS {year}: OK")
            else:
                print(f"  ACS {year}: HTTP {resp.status_code}")

            time.sleep(0.5)
        except Exception as e:
            print(f"  ACS {year} error: {e}")
            time.sleep(1)

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(os.path.join(OUTPUT_DIR, 'census_acs.csv'), index=False)
        print(f"  Saved {len(df)} Census ACS records")
        return df
    else:
        print("  Census download failed — creating placeholders")
        create_placeholder('median_household_income', 'median_household_income',
                           'Census ACS Table B19013')
        create_placeholder('population', 'population',
                           'Census annual population estimates')
        return None


def download_census_demographics():
    """
    Download age and race demographics from ACS.
    B09021 or C18120 for age; B03002 for race/ethnicity.
    """
    print("Downloading Census demographics (age 65+, race)...")

    all_data = []

    for year in range(2005, 2024):
        # B01001_020E through B01001_025E = male 65+
        # B01001_044E through B01001_049E = female 65+
        # B01003_001E = total pop
        # B03002_003E = non-Hispanic white alone
        # B03002_001E = total for race table

        age_vars_male = [f'B01001_{str(i).zfill(3)}E' for i in range(20, 26)]
        age_vars_female = [f'B01001_{str(i).zfill(3)}E' for i in range(44, 50)]
        race_vars = ['B03002_003E', 'B03002_001E']
        pop_var = ['B01003_001E']

        all_vars = age_vars_male + age_vars_female + race_vars + pop_var
        var_string = ','.join(all_vars)

        url = (
            f"https://api.census.gov/data/{year}/acs/acs1"
            f"?get=NAME,{var_string}&for=state:*"
        )

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                headers = data[0]
                for row in data[1:]:
                    row_dict = dict(zip(headers, row))
                    fips = row_dict.get('state', '')
                    state = FIPS_TO_STATE.get(fips, '')
                    if not state:
                        continue

                    def safe_float(key):
                        v = row_dict.get(key)
                        try:
                            return float(v) if v and v != 'null' else 0
                        except (ValueError, TypeError):
                            return 0

                    pop = safe_float('B01003_001E')
                    age65_male = sum(safe_float(v) for v in age_vars_male)
                    age65_female = sum(safe_float(v) for v in age_vars_female)
                    age65_total = age65_male + age65_female
                    pct_65plus = (age65_total / pop * 100) if pop > 0 else np.nan

                    nh_white = safe_float('B03002_003E')
                    race_total = safe_float('B03002_001E')
                    pct_nh_white = (nh_white / race_total * 100) if race_total > 0 else np.nan

                    all_data.append({
                        'state': state,
                        'year': year,
                        'pct_65plus': round(pct_65plus, 2) if not np.isnan(pct_65plus) else np.nan,
                        'pct_nh_white': round(pct_nh_white, 2) if not np.isnan(pct_nh_white) else np.nan
                    })

                print(f"  Demographics {year}: OK")
            else:
                print(f"  Demographics {year}: HTTP {resp.status_code}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Demographics {year} error: {e}")
            time.sleep(1)

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(os.path.join(OUTPUT_DIR, 'census_demographics.csv'), index=False)
        print(f"  Saved {len(df)} demographics records")
        return df
    else:
        print("  Demographics download failed — creating placeholders")
        create_placeholder('pct_65plus', 'pct_65plus', 'Census ACS age tables')
        create_placeholder('pct_nh_white', 'pct_nh_white', 'Census ACS race/ethnicity')
        return None


def download_census_poverty_uninsurance():
    """Download poverty rate and uninsurance rate from ACS subject tables."""
    print("Downloading poverty rate and uninsurance rate...")

    all_data = []

    for year in range(2008, 2024):
        # Use detailed tables instead of subject tables (more reliable without key)
        # B17001_001E = total for poverty status determination
        # B17001_002E = income below poverty level
        # B27010_001E = total for insurance status (civilian noninstitutionalized)
        # B27010_017E = no health insurance (one of the uninsured categories)
        # Actually, let's use B27001 which is cleaner:
        # B27001_001E = total civilian noninst. pop
        # We need to sum the "no health insurance" subcategories

        # Simpler approach: use S2701 from subject tables
        # But those may not work without a key. Let's try B-tables:
        # B17001_002E / B17001_001E = poverty rate
        poverty_vars = ['B17001_001E', 'B17001_002E']

        var_string = ','.join(poverty_vars)
        url = (
            f"https://api.census.gov/data/{year}/acs/acs1"
            f"?get=NAME,{var_string}&for=state:*"
        )

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                headers = data[0]
                for row in data[1:]:
                    row_dict = dict(zip(headers, row))
                    fips = row_dict.get('state', '')
                    state = FIPS_TO_STATE.get(fips, '')
                    if not state:
                        continue

                    def safe_float(key):
                        v = row_dict.get(key)
                        try:
                            return float(v) if v and v != 'null' else 0
                        except (ValueError, TypeError):
                            return 0

                    pov_total = safe_float('B17001_001E')
                    pov_below = safe_float('B17001_002E')
                    poverty_rate = (pov_below / pov_total * 100) if pov_total > 0 else np.nan

                    all_data.append({
                        'state': state,
                        'year': year,
                        'poverty_rate': round(poverty_rate, 2) if not np.isnan(poverty_rate) else np.nan,
                    })

                print(f"  Poverty {year}: OK")
            else:
                print(f"  Poverty {year}: HTTP {resp.status_code}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Poverty {year} error: {e}")
            time.sleep(1)

    # Now try uninsurance rate from B27010 or B27001
    for year in range(2008, 2024):
        # B27001: age by health insurance coverage
        # Uninsured counts are scattered across age groups
        # Simpler: use DP03 (selected economic characteristics) or
        # just B27010_017E + B27010_033E + B27010_050E + B27010_066E (uninsured by age group)
        # Actually let's use the simple approach: B27010
        unins_vars = ['B27010_017E', 'B27010_033E', 'B27010_050E', 'B27010_066E', 'B27010_001E']
        var_string = ','.join(unins_vars)
        url = (
            f"https://api.census.gov/data/{year}/acs/acs1"
            f"?get=NAME,{var_string}&for=state:*"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                headers = data[0]
                for row in data[1:]:
                    row_dict = dict(zip(headers, row))
                    fips = row_dict.get('state', '')
                    state = FIPS_TO_STATE.get(fips, '')
                    if not state:
                        continue

                    def safe_float(key):
                        v = row_dict.get(key)
                        try:
                            return float(v) if v and v != 'null' else 0
                        except (ValueError, TypeError):
                            return 0

                    total = safe_float('B27010_001E')
                    uninsured = sum(safe_float(v) for v in ['B27010_017E', 'B27010_033E',
                                                            'B27010_050E', 'B27010_066E'])
                    unins_rate = (uninsured / total * 100) if total > 0 else np.nan

                    # Find matching record and add uninsurance
                    for rec in all_data:
                        if rec['state'] == state and rec['year'] == year:
                            rec['uninsurance_rate'] = round(unins_rate, 2) if not np.isnan(unins_rate) else np.nan
                            break

            time.sleep(0.5)
        except Exception as e:
            print(f"  Uninsurance {year} error: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(os.path.join(OUTPUT_DIR, 'poverty_uninsurance.csv'), index=False)
        print(f"  Saved {len(df)} poverty/uninsurance records")
        return df
    else:
        print("  Poverty/uninsurance download failed — creating placeholders")
        create_placeholder('poverty_rate', 'poverty_rate', 'Census ACS Table S1701')
        create_placeholder('uninsurance_rate', 'uninsurance_rate', 'Census ACS Table S2701 or KFF')
        return None


def create_medicaid_expansion():
    """
    Create Medicaid expansion status by state-year.
    Hardcoded from KFF data — this is a key covariate.
    Year = first full year of expansion.
    """
    print("Creating Medicaid expansion data (hardcoded from KFF)...")

    # States that expanded Medicaid and the year expansion took effect
    # Source: KFF Status of State Medicaid Expansion Decisions
    expansion_year = {
        'AZ': 2014, 'AR': 2014, 'CA': 2014, 'CO': 2014, 'CT': 2014,
        'DE': 2014, 'DC': 2014, 'HI': 2014, 'IL': 2014, 'IA': 2014,
        'KY': 2014, 'MD': 2014, 'MA': 2014, 'MI': 2014, 'MN': 2014,
        'NV': 2014, 'NJ': 2014, 'NM': 2014, 'NY': 2014, 'ND': 2014,
        'OH': 2014, 'OR': 2014, 'RI': 2014, 'VT': 2014, 'WA': 2014,
        'WV': 2014,
        'NH': 2014, 'PA': 2015, 'IN': 2015, 'AK': 2015,
        'MT': 2016, 'LA': 2016,
        'VA': 2019, 'ME': 2019, 'ID': 2020,
        'NE': 2020, 'UT': 2020,
        'OK': 2021, 'MO': 2021,
        'SD': 2024, 'NC': 2024,
    }
    # States that have NOT expanded as of 2023:
    # AL, FL, GA, KS, MS, SC, TN, TX, WI, WY

    rows = []
    for state in ALL_STATES:
        for year in YEARS:
            exp_year = expansion_year.get(state)
            expanded = 1 if (exp_year is not None and year >= exp_year) else 0
            rows.append({
                'state': state,
                'year': year,
                'medicaid_expansion': expanded
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'medicaid_expansion.csv'), index=False)
    print(f"  Saved {len(df)} Medicaid expansion records")
    return df


def create_marketplace_type():
    """
    Create state marketplace type (state-run vs federal vs partnership).
    Hardcoded from KFF data.
    """
    print("Creating marketplace type data (hardcoded from KFF)...")

    # State-based marketplaces (as of 2023 — simplified, some changed over time)
    # 0 = federally facilitated (FFM), 1 = state-based (SBM), 2 = state-partnership
    # Simplified: we'll mark states that had SBMs from 2014+
    state_based = {
        'CA', 'CO', 'CT', 'DC', 'ID', 'KY', 'MA', 'MD', 'MN',
        'NV', 'NJ', 'NM', 'NY', 'PA', 'RI', 'VT', 'WA'
    }
    # Most others use FFM (healthcare.gov)

    rows = []
    for state in ALL_STATES:
        for year in YEARS:
            if year >= 2014:
                mtype = 'SBM' if state in state_based else 'FFM'
            else:
                mtype = 'pre_ACA'
            rows.append({
                'state': state,
                'year': year,
                'marketplace_type': mtype
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'marketplace_type.csv'), index=False)
    print(f"  Saved {len(df)} marketplace records")
    return df


def create_pre_aca_guaranteed_issue():
    """
    Pre-ACA guaranteed issue / community rating laws.
    Time-invariant state characteristic (before ACA mandated it nationally in 2014).
    """
    print("Creating pre-ACA guaranteed issue data...")

    # States with pre-ACA individual market guaranteed issue or community rating
    # Source: literature, NAIC
    pre_aca_gi = {
        'MA': 1,  # Full guaranteed issue + community rating since 2006
        'NY': 1,  # Guaranteed issue since 1993, community rating
        'NJ': 1,  # Individual and small group guaranteed issue since 1993
        'VT': 1,  # Guaranteed issue since 1992, community rating
        'ME': 1,  # Guaranteed issue since 1993 (modified)
        'WA': 1,  # Guaranteed issue 1995-2000 (partially repealed)
    }

    rows = []
    for state in ALL_STATES:
        rows.append({
            'state': state,
            'pre_aca_guaranteed_issue': pre_aca_gi.get(state, 0)
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'pre_aca_guaranteed_issue.csv'), index=False)
    print(f"  Saved {len(df)} pre-ACA guaranteed issue records")
    return df


def create_placeholder(variable_name, column_name, source_note):
    """Create a placeholder CSV with correct structure."""
    rows = []
    for state in ALL_STATES:
        for year in YEARS:
            rows.append({'state': state, 'year': year, column_name: np.nan})

    df = pd.DataFrame(rows)
    filepath = os.path.join(OUTPUT_DIR, f'{variable_name}.csv')
    # Add source comment at top
    with open(filepath, 'w') as f:
        f.write(f"# TODO: Download from {source_note}\n")
    df.to_csv(filepath, mode='a', index=False)
    print(f"  Created placeholder: {variable_name}.csv")
    return df


def create_healthcare_spending_placeholder():
    """Placeholder for CMS State Health Expenditure data."""
    print("Creating healthcare spending placeholder...")
    return create_placeholder(
        'healthcare_spending_per_capita',
        'healthcare_spending_per_capita',
        'CMS State Health Expenditure Accounts: https://www.cms.gov/data-research/statistics-trends-and-reports/national-health-expenditure-data/state-residence'
    )


if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 1: Downloading/Creating State-Year Covariates")
    print("=" * 60)

    # Hardcoded data (always works)
    create_medicaid_expansion()
    create_marketplace_type()
    create_pre_aca_guaranteed_issue()

    # API downloads (may fail without keys / rate limits)
    download_bls_unemployment()
    download_census_acs()
    download_census_demographics()
    download_census_poverty_uninsurance()

    # Placeholders for manual data
    create_healthcare_spending_placeholder()

    print("\n" + "=" * 60)
    print("Done! Check Datasets/covariates/ for output files.")
    print("Files with '# TODO' headers need manual data entry.")
    print("=" * 60)

    # List what we got
    for f in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  {f}: {size:,} bytes")
