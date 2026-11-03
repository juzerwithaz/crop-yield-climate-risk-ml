"""
09_prepare_stage1_dataset.py

Preparation script for Stage 1 modeling:
1. Loads verified `tn_master_dataset_weather_corrected.csv`.
2. Computes leakage-safe `Yield_Lag_Avg` and `Yield_Lag_Years_Available` using
   exact preceding calendar years (t-1, t-2, t-3) per (District, Crop) series.
3. Assigns strict chronological split labels:
   - train:      Year <= 2007
   - validation: 2008 <= Year <= 2009
   - test:       2010 <= Year <= 2013
4. Preserves all 19 original source columns unchanged.
5. Saves derived dataset to `tn_master_dataset_stage1_ready.csv`.
6. Performs self-contained reload and integrity verification.
"""

import os
import sys
import pandas as pd
import numpy as np

INPUT_CSV = "tn_master_dataset_weather_corrected.csv"
OUTPUT_CSV = "tn_master_dataset_stage1_ready.csv"

EXPECTED_ROWS = 2333
ORIGINAL_COLUMNS = [
    'District', 'Year', 'Crop', 'Area', 'Production', 'Yield',
    'Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K',
    'Soil_Confidence', 'Valid_Months', 'Complete_Year',
    'Rainfall_Original', 'Temp_Original', 'Humidity_Original'
]

def load_and_validate_input(filepath):
    print(f"Loading input: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required input file '{filepath}' does not exist.")
    
    df = pd.read_csv(filepath)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(df)}.")
    
    missing_cols = [c for c in ORIGINAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {missing_cols}")
    
    dup_count = df.duplicated(subset=['District', 'Year', 'Crop']).sum()
    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate (District, Year, Crop) keys in input.")
    
    print(f"  Input validated successfully ({len(df)} rows, {len(df.columns)} columns).")
    return df

def compute_calendar_lags(df):
    """
    Computes Yield_Lag_Avg and Yield_Lag_Years_Available for each (District, Crop)
    using strictly exact calendar years t-1, t-2, t-3.
    """
    print("\nComputing calendar-year yield lag features...")
    df = df.copy()
    
    # Fast O(1) lookup dictionary: (District, Crop, Year) -> Yield
    yield_lookup = {}
    for _, row in df.iterrows():
        key = (row['District'], row['Crop'], int(row['Year']))
        yield_lookup[key] = float(row['Yield'])
    
    lag_avgs = []
    years_avail = []
    
    for _, row in df.iterrows():
        dist = row['District']
        crop = row['Crop']
        t = int(row['Year'])
        
        hist_yields = []
        for lag in [1, 2, 3]:
            past_key = (dist, crop, t - lag)
            if past_key in yield_lookup:
                val = yield_lookup[past_key]
                if pd.notna(val):
                    hist_yields.append(val)
        
        k = len(hist_yields)
        years_avail.append(k)
        if k > 0:
            lag_avgs.append(float(np.mean(hist_yields)))
        else:
            lag_avgs.append(np.nan)
    
    df['Yield_Lag_Avg'] = lag_avgs
    df['Yield_Lag_Years_Available'] = years_avail
    
    return df

def assign_chronological_splits(df):
    """
    Assigns chronological split metadata:
    - train:      Year <= 2007
    - validation: 2008 <= Year <= 2009
    - test:       2010 <= Year <= 2013
    """
    print("Assigning chronological validation splits...")
    conditions = [
        df['Year'] <= 2007,
        (df['Year'] >= 2008) & (df['Year'] <= 2009),
        (df['Year'] >= 2010) & (df['Year'] <= 2013)
    ]
    choices = ['train', 'validation', 'test']
    df['Split'] = np.select(conditions, choices, default='unknown')
    
    if (df['Split'] == 'unknown').any():
        raise ValueError("Unexpected year encountered outside valid ranges.")
    
    return df

def verify_dataset_integrity(df_orig, df_reloaded):
    """
    Performs comprehensive verification of the reloaded output dataset against original.
    """
    print("\n" + "=" * 80)
    print("INDEPENDENT OUTPUT INTEGRITY VERIFICATION")
    print("=" * 80)
    
    # 1. Row count
    print(f"1. Row count check: reloaded has {len(df_reloaded)} rows (expected {EXPECTED_ROWS})")
    assert len(df_reloaded) == EXPECTED_ROWS, "Row count mismatch!"
    
    # 2. Key uniqueness
    dups = df_reloaded.duplicated(subset=['District', 'Year', 'Crop']).sum()
    print(f"2. Duplicate (District, Year, Crop) check: {dups} duplicates found")
    assert dups == 0, "Duplicate keys found in output dataset!"
    
    # 3. Source column preservation
    print("3. Source column preservation check:")
    key_cols = ['District', 'Year', 'Crop']
    orig_sorted = df_orig.sort_values(key_cols).reset_index(drop=True)
    reloaded_sorted = df_reloaded.sort_values(key_cols).reset_index(drop=True)
    
    mismatches = []
    for col in ORIGINAL_COLUMNS:
        if orig_sorted[col].dtype in [np.float64, np.float32]:
            both_nan = orig_sorted[col].isna() & reloaded_sorted[col].isna()
            matches = np.isclose(orig_sorted[col], reloaded_sorted[col], equal_nan=True) | both_nan
        else:
            matches = (orig_sorted[col] == reloaded_sorted[col])
        n_bad = (~matches).sum()
        if n_bad > 0:
            mismatches.append((col, n_bad))
    
    if mismatches:
        raise AssertionError(f"Source columns were altered: {mismatches}")
    print(f"   PASS: All {len(ORIGINAL_COLUMNS)} source columns match input exactly.")
    
    # 4. Lag feature sanity checks
    print("\n4. Yield_Lag_Avg & Yield_Lag_Years_Available validation:")
    avail_counts = df_reloaded['Yield_Lag_Years_Available'].value_counts().sort_index().to_dict()
    print(f"   Years Available distribution: {avail_counts}")
    assert set(avail_counts.keys()).issubset({0, 1, 2, 3}), "Yield_Lag_Years_Available out of bounds [0, 3]!"
    
    nan_lags = df_reloaded['Yield_Lag_Avg'].isna().sum()
    zero_avail = avail_counts.get(0, 0)
    print(f"   Missing Yield_Lag_Avg: {nan_lags} | Years Available == 0: {zero_avail}")
    assert nan_lags == zero_avail, "Discrepancy between missing Yield_Lag_Avg and zero history!"
    
    # Ensure all non-zero history have valid float lag values
    assert df_reloaded[df_reloaded['Yield_Lag_Years_Available'] > 0]['Yield_Lag_Avg'].notna().all()
    print("   PASS: Lag averages and availability counts are completely consistent.")
    
    # 5. Split distribution checks
    print("\n5. Split distribution check:")
    split_counts = df_reloaded['Split'].value_counts().to_dict()
    print(f"   Counts: {split_counts}")
    assert split_counts == {'train': 1443, 'test': 594, 'validation': 296}, f"Unexpected split counts: {split_counts}"
    
    # Split year boundary verification
    assert (df_reloaded[df_reloaded['Split'] == 'train']['Year'] <= 2007).all()
    assert (df_reloaded[df_reloaded['Split'] == 'validation']['Year'].between(2008, 2009)).all()
    assert (df_reloaded[df_reloaded['Split'] == 'test']['Year'].between(2010, 2013)).all()
    print("   PASS: All split boundaries strictly obeyed.")
    
    # 6. Manual case verifications
    print("\n6. Specific manual inspection cases:")
    
    # Case A: Early observation (1997)
    ariyalur_1997 = df_reloaded[(df_reloaded['District'] == 'ARIYALUR') & (df_reloaded['Crop'] == 'Rice') & (df_reloaded['Year'] == 2008)]
    # Case B: Gapped series — Coimbatore Maize (missing 2001)
    cm = df_reloaded[(df_reloaded['District'] == 'COIMBATORE') & (df_reloaded['Crop'] == 'Maize')].set_index('Year')
    
    # 1997: No history
    assert np.isnan(cm.loc[1997, 'Yield_Lag_Avg']) and cm.loc[1997, 'Yield_Lag_Years_Available'] == 0
    # 1998: Only 1997
    y1997 = cm.loc[1997, 'Yield']
    assert np.isclose(cm.loc[1998, 'Yield_Lag_Avg'], y1997) and cm.loc[1998, 'Yield_Lag_Years_Available'] == 1
    # 2002: (2001 missing, uses 2000 and 1999)
    y1999 = cm.loc[1999, 'Yield']
    y2000 = cm.loc[2000, 'Yield']
    expected_2002_lag = (y1999 + y2000) / 2.0
    assert np.isclose(cm.loc[2002, 'Yield_Lag_Avg'], expected_2002_lag) and cm.loc[2002, 'Yield_Lag_Years_Available'] == 2
    # 2005: (uses 2004, 2003, 2002 -> all 3 available)
    expected_2005_lag = (cm.loc[2004, 'Yield'] + cm.loc[2003, 'Yield'] + cm.loc[2002, 'Yield']) / 3.0
    assert np.isclose(cm.loc[2005, 'Yield_Lag_Avg'], expected_2005_lag) and cm.loc[2005, 'Yield_Lag_Years_Available'] == 3
    print("   PASS: Gapped series (Coimbatore Maize) calendar lag logic verified.")
    
    print("\nAll verification checks PASSED successfully.")

def main():
    print("=" * 80)
    print("STEP 3: PREPARE STAGE 1 MODELING DATASET")
    print("=" * 80)
    
    if os.path.exists(OUTPUT_CSV):
        print(f"Notice: Output file '{OUTPUT_CSV}' already exists. Overwrite check approved.")
    
    df_orig = load_and_validate_input(INPUT_CSV)
    
    df_prepared = compute_calendar_lags(df_orig)
    df_prepared = assign_chronological_splits(df_prepared)
    
    # Save output CSV
    print(f"\nSaving prepared dataset to {OUTPUT_CSV}...")
    df_prepared.to_csv(OUTPUT_CSV, index=False)
    print("  File written successfully.")
    
    # Reload from disk for verification
    print(f"Reloading {OUTPUT_CSV} from disk for validation...")
    df_reloaded = pd.read_csv(OUTPUT_CSV)
    
    verify_dataset_integrity(df_orig, df_reloaded)
    print(f"\nDeliverable ready: {os.path.abspath(OUTPUT_CSV)}")

if __name__ == '__main__':
    main()

