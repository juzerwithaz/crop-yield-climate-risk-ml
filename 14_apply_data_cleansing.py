"""
14_apply_data_cleansing.py

Versioned Data-Cleansing & Lineage Generation Script for Stage 1.
Implements Document ID: SPEC-CROP-PROD-2026-V2.3.

Core Transformations:
1. Reconciles raw crop records from `tn_crop_yield_filtered.csv` with strict one-to-one merge validation.
2. Preserves explicit multi-tier lineage:
   - Source Lineage: Area_Source, Production_Source, Yield_Source, Source_Records_Count, Source_Missing_Seasons_Count, Source_Seasons.
   - V1 Lineage: Area_V1, Production_V1, Yield_V1.
   - Cleansed Active: Area, Production, Yield.
3. Corrects missing-production classification:
   - 35 records with pre-cleansing Yield_V1 == 0.0 are classified as EXCLUDED_SOURCE_MISSING_PRODUCTION.
   - Active Production and Yield are reset to NaN.
   - Documented as an aggregation artifact from sum(min_count=0); no genuine source zero existed.
4. Preserves 5 Coimbatore 2008 records as EXCLUDED_ADMIN_BIFURCATION (composite production on residual area).
5. Scales 1997 Sugarcane production and yield by 0.10 (Strongly Supported 10x unit scale discrepancy).
6. Preserves partial-season missingness metadata for (TUTICORIN, 2013, Maize) as PARTIAL_SEASON_MISSING_PRODUCTION.
7. Recalculates leakage-safe calendar lags (t-1, t-2, t-3) using only eligible yields in lookup.
8. Computes training-only risk thresholds (tau_20) on eligible training data.
9. Runs comprehensive assertion validation before writing `tn_master_dataset_stage1_ready_v2.csv`.
"""

import os
import sys
import numpy as np
import pandas as pd

SOURCE_RAW_CSV = "tn_crop_yield_filtered.csv"
INPUT_V1_CSV = "tn_master_dataset_stage1_ready.csv"
OUTPUT_V2_CSV = "tn_master_dataset_stage1_ready_v2.csv"
EXPECTED_ROWS = 2333

FLAG_VALID = 'VALID'
FLAG_CLEANSED_UNIT_SCALED = 'CLEANSED_UNIT_SCALED'
FLAG_EXCLUDED_SOURCE_MISSING = 'EXCLUDED_SOURCE_MISSING_PRODUCTION'
FLAG_EXCLUDED_ADMIN_BIFURCATION = 'EXCLUDED_ADMIN_BIFURCATION'
ELIGIBLE_PRIMARY_FLAGS = [FLAG_VALID, FLAG_CLEANSED_UNIT_SCALED]

STATUS_COMPLETE = 'COMPLETE_SEASONS'
STATUS_PARTIAL_MISSING = 'PARTIAL_SEASON_MISSING_PRODUCTION'
STATUS_ALL_MISSING = 'ALL_SEASONS_MISSING_PRODUCTION'

def series_equal_with_nan(s1, s2):
    """Missing-value-aware series comparison: numerical equality or identical NaN pattern."""
    return ((s1 == s2) | (s1.isna() & s2.isna())).all()

def float_series_close_with_nan(s1, s2, rtol=1e-7, atol=1e-7):
    """Missing-value-aware float comparison within numerical tolerance."""
    both_nan = s1.isna() & s2.isna()
    close_vals = np.isclose(s1.fillna(0), s2.fillna(0), rtol=rtol, atol=atol)
    return (both_nan | close_vals).all()

def run_pre_write_assertions(df_v2, df_v1):
    print("\n" + "=" * 80)
    print("EXECUTING MANDATORY PRE-WRITE VALIDATION ASSERTIONS")
    print("=" * 80)

    # 1. Dataset Dimensions & Natural Keys
    print("Assertion 1: Row count and natural key uniqueness...")
    assert len(df_v2) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df_v2)}"
    dup_keys = df_v2.duplicated(subset=['District', 'Crop', 'Year']).sum()
    assert dup_keys == 0, f"Found {dup_keys} duplicate natural keys!"
    assert (df_v2['_source_merge'] == 'both').all(), "Merge indicator incomplete: unmatched keys exist!"
    print("  PASS: Exactly 2,333 rows, 0 duplicate keys, 100% bijective merge.")

    # 2. Central Quality Status Distribution
    print("Assertion 2: Central Data_Quality_Flag distribution...")
    flag_counts = df_v2['Data_Quality_Flag'].value_counts().to_dict()
    expected_flags = {
        FLAG_VALID: 2266,
        FLAG_EXCLUDED_SOURCE_MISSING: 35,
        FLAG_CLEANSED_UNIT_SCALED: 27,
        FLAG_EXCLUDED_ADMIN_BIFURCATION: 5
    }
    assert flag_counts == expected_flags, f"Flag counts mismatch: got {flag_counts}, expected {expected_flags}"
    print(f"  PASS: Exact quality flag counts verified: {flag_counts}")

    # 3. Seasonal Aggregation Status Distribution
    print("Assertion 3: Seasonal_Aggregation_Status distribution...")
    season_counts = df_v2['Seasonal_Aggregation_Status'].value_counts().to_dict()
    expected_seasons = {
        STATUS_COMPLETE: 2297,
        STATUS_ALL_MISSING: 35,
        STATUS_PARTIAL_MISSING: 1
    }
    assert season_counts == expected_seasons, f"Seasonal status mismatch: got {season_counts}, expected {expected_seasons}"
    print(f"  PASS: Exact seasonal status counts verified: {season_counts}")

    # 4. Partial-Season Flag Verification (Tuticorin 2013 Maize)
    print("Assertion 4: Partial-season missingness metadata (Tuticorin 2013 Maize)...")
    tut = df_v2[(df_v2['District'] == 'TUTICORIN') & (df_v2['Crop'] == 'Maize') & (df_v2['Year'] == 2013)].iloc[0]
    assert tut['Seasonal_Aggregation_Status'] == STATUS_PARTIAL_MISSING
    assert tut['Data_Quality_Flag'] == FLAG_VALID
    assert tut['Source_Missing_Seasons_Count'] == 1
    assert tut['Source_Records_Count'] == 2
    assert tut['Area'] == 37537 and np.isclose(tut['Yield'], 2.078909, atol=1e-4)
    print(f"  PASS: Tuticorin 2013 Maize verified: {tut['Source_Seasons']} with 1 missing season, preserved valid yield {tut['Yield']:.4f} t/ha.")

    # 5. Pre-Cleansing V1 Value Reconciliation
    print("Assertion 5: Pre-cleansing V1 value preservation...")
    assert series_equal_with_nan(df_v2['Area_V1'], df_v1['Area']), "Area_V1 altered!"
    assert series_equal_with_nan(df_v2['Production_V1'], df_v1['Production']), "Production_V1 altered!"
    assert float_series_close_with_nan(df_v2['Yield_V1'], df_v1['Yield']), "Yield_V1 altered!"
    for col in ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K',
                'Soil_Confidence', 'Valid_Months', 'Complete_Year', 'Split']:
        assert series_equal_with_nan(df_v2[col], df_v1[col]), f"Source feature {col} altered!"
    print("  PASS: All pre-cleansing V1 values and environmental features preserved byte-for-byte.")

    # 6. Source-to-V1 Reconciliations & Missingness Verification
    print("Assertion 6: Source-to-V1 reconciliations and missingness verification...")
    # Area reconciles across ALL 2,333 records
    assert (df_v2['Area_Source'] == df_v2['Area_V1']).all(), "Source Area does not match V1 Area!"
    # Valid records: Production and Yield match Source exactly
    valid_mask = (df_v2['Data_Quality_Flag'] == FLAG_VALID)
    assert (df_v2.loc[valid_mask, 'Production_Source'] == df_v2.loc[valid_mask, 'Production_V1']).all()
    assert float_series_close_with_nan(df_v2.loc[valid_mask, 'Yield_Source'], df_v2.loc[valid_mask, 'Yield_V1'])

    # Missing production records: Source is NaN, V1 was 0.0, active is NaN
    missing_mask = (df_v2['Data_Quality_Flag'] == FLAG_EXCLUDED_SOURCE_MISSING)
    assert len(df_v2[missing_mask]) == 35
    assert df_v2.loc[missing_mask, 'Production_Source'].isna().all(), "Source production was not NaN!"
    assert df_v2.loc[missing_mask, 'Yield_Source'].isna().all(), "Source yield was not NaN!"
    assert (df_v2.loc[missing_mask, 'Production_V1'] == 0.0).all(), "V1 production was not 0.0!"
    assert (df_v2.loc[missing_mask, 'Yield_V1'] == 0.0).all(), "V1 yield was not 0.0!"
    assert df_v2.loc[missing_mask, 'Production'].isna().all(), "Active production was not NaN!"
    assert df_v2.loc[missing_mask, 'Yield'].isna().all(), "Active yield was not NaN!"
    print("  PASS: 35 missing production records verified: source NaN -> V1 0.0 artifact -> active NaN.")

    # 7. 1997 Sugarcane Yield Assertions
    print("Assertion 7: 1997 Sugarcane yield metrics (Area-Weighted vs. Unweighted Mean)...")
    sc_1997 = df_v2[(df_v2['Crop'] == 'Sugarcane') & (df_v2['Year'] == 1997)]
    assert len(sc_1997) == 27
    assert (sc_1997['Data_Quality_Flag'] == FLAG_CLEANSED_UNIT_SCALED).all()
    # Area-weighted statewide yield == 108.5467 t/ha
    state_weighted_yield = sc_1997['Production'].sum() / sc_1997['Area'].sum()
    assert np.isclose(state_weighted_yield, 108.546731, atol=1e-4), f"Area-weighted yield mismatch: {state_weighted_yield}"
    # Unweighted district mean yield == 113.3472 t/ha
    district_mean_yield = sc_1997['Yield'].mean()
    assert np.isclose(district_mean_yield, 113.347193, atol=1e-4), f"District mean yield mismatch: {district_mean_yield}"
    # District outliers preserved
    assert np.isclose(sc_1997['Yield'].min(), 62.3331, atol=1e-3), "Min yield mismatch (Thiruvarur)"
    assert np.isclose(sc_1997['Yield'].max(), 224.7533, atol=1e-3), "Max yield mismatch (Madurai)"
    print(f"  PASS: Area-weighted yield = {state_weighted_yield:.2f} t/ha, Unweighted mean = {district_mean_yield:.2f} t/ha, Range = [{sc_1997['Yield'].min():.2f}, {sc_1997['Yield'].max():.2f}].")

    # 8. Coimbatore 2008 Five-Crop Exclusion
    print("Assertion 8: Coimbatore 2008 five-crop administrative exclusion...")
    cb_2008 = df_v2[(df_v2['District'] == 'COIMBATORE') & (df_v2['Year'] == 2008)]
    assert len(cb_2008) == 5, f"Expected 5 Coimbatore 2008 records, got {len(cb_2008)}"
    assert (cb_2008['Data_Quality_Flag'] == FLAG_EXCLUDED_ADMIN_BIFURCATION).all()
    assert (cb_2008['Split'] == 'validation').all()
    print("  PASS: All 5 Coimbatore 2008 crop records cleanly excluded from primary modeling.")

    # 9. Risk Threshold & Label Assertions
    print("Assertion 9: Risk threshold recalculation and label eligibility...")
    sc_train = df_v2[(df_v2['Crop'] == 'Sugarcane') & (df_v2['Split'] == 'train') & (df_v2['Data_Quality_Flag'].isin(ELIGIBLE_PRIMARY_FLAGS))]
    assert len(sc_train) == 300, f"Expected 300 eligible Sugarcane train rows, got {len(sc_train)}"
    sc_t20 = float(round(np.percentile(sc_train['Yield'], 20), 4))
    assert np.isclose(sc_t20, 90.7270, atol=1e-4), f"Expected Sugarcane tau_20 == 90.7270, got {sc_t20}"
    # Eligible rows have not-null risk labels; excluded rows have NaN risk labels
    assert df_v2[df_v2['Data_Quality_Flag'].isin(ELIGIBLE_PRIMARY_FLAGS)]['High_Yield_Risk'].notna().all()
    assert df_v2[~df_v2['Data_Quality_Flag'].isin(ELIGIBLE_PRIMARY_FLAGS)]['High_Yield_Risk'].isna().all()
    print(f"  PASS: Sugarcane tau_20 = {sc_t20:.4f} t/ha on N=300; binary labels assigned to eligible rows; NaN for excluded.")

    # 10. Calendar Lag Non-Leaking Assertions
    print("Assertion 10: Leakage-safe calendar lag features...")
    cm_2009 = df_v2[(df_v2['District'] == 'COIMBATORE') & (df_v2['Crop'] == 'Maize') & (df_v2['Year'] == 2009)].iloc[0]
    assert np.isclose(cm_2009['Yield_Lag_Avg'], 5.331146, atol=1e-4), f"Coimbatore 2009 lag contaminated: {cm_2009['Yield_Lag_Avg']}"
    assert cm_2009['Yield_Lag_Years_Available'] == 2
    print(f"  PASS: Coimbatore 2009 lag = {cm_2009['Yield_Lag_Avg']:.4f} t/ha (cleanly skipping excluded 2008).")

    print("\n" + "=" * 80)
    print("ALL 10 PRE-WRITE INTEGRITY ASSERTIONS PASSED WITH ZERO ERRORS.")
    print("=" * 80)

def main():
    print("=" * 80)
    print("STAGE 1: GENERATE VERSIONED CLEANSED DATASET (V2)")
    print("Specification: SPEC-CROP-PROD-2026-V2.3")
    print("=" * 80)

    # 1. Load raw crop source records
    print(f"\n1. Loading raw crop records from: {SOURCE_RAW_CSV}")
    df_raw = pd.read_csv(SOURCE_RAW_CSV)
    df_raw['District'] = df_raw['District_Name'].astype(str).str.strip().str.upper()
    df_raw['Year'] = df_raw['Crop_Year'].astype(int)
    print(f"   Loaded {len(df_raw)} source rows across {df_raw['District'].nunique()} districts.")

    # Aggregate to (District, Year, Crop) with strict min_count=1
    print("   Aggregating raw seasonal records to natural keys (District, Year, Crop)...")
    grouped_raw = df_raw.groupby(['District', 'Year', 'Crop']).agg(
        Area_Source=('Area', 'sum'),
        Production_Source=('Production', lambda s: s.sum(min_count=1)),
        Source_Records_Count=('Area', 'count'),
        Source_Missing_Seasons_Count=('Production', lambda s: s.isna().sum()),
        Source_Seasons=('Season', lambda s: '; '.join(s.str.strip()))
    ).reset_index()

    grouped_raw['Yield_Source'] = np.where(
        grouped_raw['Area_Source'] > 0,
        grouped_raw['Production_Source'] / grouped_raw['Area_Source'],
        np.nan
    )

    cond_season = [
        grouped_raw['Source_Missing_Seasons_Count'] == 0,
        (grouped_raw['Source_Missing_Seasons_Count'] > 0) & (grouped_raw['Source_Missing_Seasons_Count'] < grouped_raw['Source_Records_Count']),
        grouped_raw['Source_Missing_Seasons_Count'] == grouped_raw['Source_Records_Count']
    ]
    choices_season = [STATUS_COMPLETE, STATUS_PARTIAL_MISSING, STATUS_ALL_MISSING]
    grouped_raw['Seasonal_Aggregation_Status'] = np.select(cond_season, choices_season, default='UNKNOWN')
    print(f"   Aggregated to {len(grouped_raw)} natural groups.")

    # 2. Load pre-cleansing Stage 1 v1 dataset
    print(f"\n2. Loading pre-cleansing Stage 1 dataset from: {INPUT_V1_CSV}")
    df_v1 = pd.read_csv(INPUT_V1_CSV)
    assert len(df_v1) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df_v1)}"
    print(f"   Loaded {len(df_v1)} rows from V1.")

    # 3. Merge source lineage with v1 using explicit one-to-one validation
    print("\n3. Merging source lineage with V1 using validate='one_to_one'...")
    df_v2 = pd.merge(
        df_v1,
        grouped_raw,
        on=['District', 'Year', 'Crop'],
        how='outer',
        validate='one_to_one',
        indicator='_source_merge'
    )
    assert (df_v2['_source_merge'] == 'both').all(), "Merge was not complete 1-to-1 match!"
    print("   Bijective merge successful: 2,333 matching rows.")

    # 4. Preserve pre-cleansing v1 fields
    print("\n4. Preserving pre-cleansing V1 lineage fields...")
    df_v2['Area_V1'] = df_v2['Area'].copy()
    df_v2['Production_V1'] = df_v2['Production'].copy()
    df_v2['Yield_V1'] = df_v2['Yield'].copy()

    # 5. Assign centralized quality flags and explicit reasons
    print("\n5. Assigning centralized quality flags and explicit documentation...")
    zero_mask = (df_v2['Yield_V1'] == 0.0)
    sc_1997_mask = (df_v2['Crop'] == 'Sugarcane') & (df_v2['Year'] == 1997)
    cb_2008_mask = (df_v2['District'] == 'COIMBATORE') & (df_v2['Year'] == 2008)

    df_v2['Data_Quality_Flag'] = FLAG_VALID
    df_v2.loc[zero_mask, 'Data_Quality_Flag'] = FLAG_EXCLUDED_SOURCE_MISSING
    df_v2.loc[sc_1997_mask, 'Data_Quality_Flag'] = FLAG_CLEANSED_UNIT_SCALED
    df_v2.loc[cb_2008_mask, 'Data_Quality_Flag'] = FLAG_EXCLUDED_ADMIN_BIFURCATION

    df_v2['Data_Quality_Reason'] = 'Standard valid observation'
    df_v2.loc[zero_mask, 'Data_Quality_Reason'] = (
        'Previous aggregation converted entirely missing source production (NaN) into 0.0 via sum(min_count=0); '
        'no genuine zero reported in source'
    )
    df_v2.loc[sc_1997_mask, 'Data_Quality_Reason'] = (
        'Strongly supported 10x production-unit scale discrepancy in 1997 Whole Year sugarcane records; scaled by 0.10'
    )
    df_v2.loc[cb_2008_mask, 'Data_Quality_Reason'] = (
        'Administrative bifurcation mismatch: undivided composite production assigned to post-bifurcation residual area'
    )

    # 6. Set active modeling fields
    print("\n6. Applying active cleansed values for modeling...")
    # Sugarcane 1997: scale by 0.10
    df_v2.loc[sc_1997_mask, 'Production'] = df_v2.loc[sc_1997_mask, 'Production_V1'] * 0.10
    df_v2.loc[sc_1997_mask, 'Yield'] = df_v2.loc[sc_1997_mask, 'Yield_V1'] * 0.10

    # Missing production records: set active modeling fields to NaN (reflecting genuine source missingness)
    df_v2.loc[zero_mask, 'Production'] = np.nan
    df_v2.loc[zero_mask, 'Yield'] = np.nan

    # 7. Compute leakage-safe calendar lags using only eligible yields in lookup
    print("\n7. Computing leakage-safe calendar lags (t-1, t-2, t-3)...")
    yield_lookup = {}
    for _, row in df_v2.iterrows():
        if row['Data_Quality_Flag'] in ELIGIBLE_PRIMARY_FLAGS:
            key = (row['District'], row['Crop'], int(row['Year']))
            yield_lookup[key] = float(row['Yield'])

    lag_avgs = []
    years_avail = []
    for _, row in df_v2.iterrows():
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
        lag_avgs.append(float(np.mean(hist_yields)) if k > 0 else np.nan)

    df_v2['Yield_Lag_Avg'] = lag_avgs
    df_v2['Yield_Lag_Years_Available'] = years_avail

    # 8. Compute training-only risk thresholds on eligible training observations
    print("\n8. Computing training-only risk thresholds (tau_20)...")
    train_eligible = df_v2[(df_v2['Split'] == 'train') & (df_v2['Data_Quality_Flag'].isin(ELIGIBLE_PRIMARY_FLAGS))]
    risk_thresholds = {}
    for crop in sorted(df_v2['Crop'].unique()):
        cy = train_eligible[train_eligible['Crop'] == crop]['Yield'].values
        t20 = float(round(np.percentile(cy, 20), 4))
        risk_thresholds[crop] = t20
        print(f"   {crop:15s}: tau_20 = {t20:.4f} t/ha (Train N = {len(cy)})")

    # 9. Assign risk labels (NaN for excluded records)
    print("\n9. Assigning High_Yield_Risk target labels...")
    df_v2['High_Yield_Risk'] = np.nan
    for crop, t20 in risk_thresholds.items():
        mask_crop = (df_v2['Crop'] == crop) & (df_v2['Data_Quality_Flag'].isin(ELIGIBLE_PRIMARY_FLAGS))
        df_v2.loc[mask_crop, 'High_Yield_Risk'] = (df_v2.loc[mask_crop, 'Yield'] <= t20).astype(int)

    # 10. Re-order columns logically
    columns_ordered = [
        'District', 'Year', 'Crop', 'Area', 'Production', 'Yield',
        'Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K',
        'Soil_Confidence', 'Valid_Months', 'Complete_Year',
        'Rainfall_Original', 'Temp_Original', 'Humidity_Original',
        'Yield_Lag_Avg', 'Yield_Lag_Years_Available',
        'Split', 'High_Yield_Risk',
        'Data_Quality_Flag', 'Data_Quality_Reason', 'Seasonal_Aggregation_Status',
        'Area_Source', 'Production_Source', 'Yield_Source',
        'Source_Records_Count', 'Source_Missing_Seasons_Count', 'Source_Seasons',
        'Area_V1', 'Production_V1', 'Yield_V1',
        '_source_merge'
    ]
    df_v2 = df_v2[columns_ordered]

    # 11. Run assertions BEFORE writing output
    run_pre_write_assertions(df_v2, df_v1)

    # 12. Save versioned output file
    print(f"\n12. Saving cleansed dataset to: {OUTPUT_V2_CSV}...")
    df_v2.to_csv(OUTPUT_V2_CSV, index=False)
    print(f"    Successfully wrote {len(df_v2)} rows to {OUTPUT_V2_CSV}.")

    # 13. Reload and perform independent re-check
    print(f"\n13. Reloading {OUTPUT_V2_CSV} for independent reload verification...")
    df_reloaded = pd.read_csv(OUTPUT_V2_CSV)
    run_pre_write_assertions(df_reloaded, df_v1)
    print("\nSTAGE 1 COMPLETE: Versioned derived dataset ready and independently verified.")

if __name__ == '__main__':
    main()

