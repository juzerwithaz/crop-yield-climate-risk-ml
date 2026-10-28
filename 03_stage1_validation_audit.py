import pandas as pd
import numpy as np
import os

def run_validation_audit(filepath="tn_master_dataset_clean.csv"):
    print("=" * 80)
    print(" 03_STAGE1_VALIDATION_AUDIT: DATA & PARTITION QUALITY REPORT")
    print("=" * 80)

    # Tracking for the final summary
    audit_flags = {
        "passed": [],
        "investigate": [],
        "unanswered": []
    }

    # -------------------------------------------------------------------------
    # 1. LOAD DATASET & BASIC INFO
    # -------------------------------------------------------------------------
    if not os.path.exists(filepath):
        print(f"ERROR: Dataset not found at {filepath}")
        return

    print("\n--- 1. DATASET OVERVIEW ---")
    df = pd.read_csv(filepath)
    
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn Names and Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")
        
    req_cols = ['District', 'Year', 'Crop', 'Area', 'Production', 'Yield']
    missing_req = [c for c in req_cols if c not in df.columns]
    if missing_req:
        print(f"\nCRITICAL ERROR: Missing required columns: {missing_req}")
        return
    else:
        audit_flags["passed"].append("All required base columns are present.")

    year_min, year_max = df['Year'].min(), df['Year'].max()
    crops = df['Crop'].unique()
    print(f"\nYear Range: {year_min} to {year_max}")
    print(f"Crops Identified ({len(crops)}): {', '.join(map(str, crops))}")

    # -------------------------------------------------------------------------
    # 2. DATA QUALITY CHECKS
    # -------------------------------------------------------------------------
    print("\n--- 2. DATA QUALITY CHECKS ---")
    
    # Duplicates
    dupes = df.duplicated(subset=['District', 'Year', 'Crop'], keep=False)
    num_dupes = dupes.sum()
    print(f"Duplicate District+Year+Crop records: {num_dupes}")
    if num_dupes > 0:
        audit_flags["investigate"].append(f"Found {num_dupes} duplicate entity-year records.")
    else:
        audit_flags["passed"].append("No duplicate District+Year+Crop records.")

    # Missing values in required columns
    print("\nMissing values in required columns:")
    missing_counts = df[req_cols].isna().sum()
    print(missing_counts.to_string())
    if missing_counts.sum() > 0:
        audit_flags["investigate"].append("Missing values present in required identification/target columns.")
    else:
        audit_flags["passed"].append("No missing values in required base columns.")

    # Non-finite numerics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_finite = (~np.isfinite(df[numeric_cols].fillna(0))).sum()
    non_finite = non_finite[non_finite > 0]
    if not non_finite.empty:
        print("\nNon-finite numeric values (Inf/-Inf):")
        print(non_finite.to_string())
        audit_flags["investigate"].append("Non-finite numeric values detected.")
    else:
        print("\nNon-finite numeric values (Inf/-Inf): 0")

    # Negatives & Zeros
    neg_area = (df['Area'] < 0).sum()
    neg_prod = (df['Production'] < 0).sum()
    neg_yield = (df['Yield'] < 0).sum()
    zero_prod = (df['Production'] == 0).sum()
    zero_yield = (df['Yield'] == 0).sum()
    
    print(f"\nNegative Values -> Area: {neg_area}, Production: {neg_prod}, Yield: {neg_yield}")
    print(f"Zero Values     -> Production: {zero_prod}, Yield: {zero_yield}")
    
    if any(x > 0 for x in [neg_area, neg_prod, neg_yield]):
        audit_flags["investigate"].append("Negative values found in Area, Production, or Yield.")
    if zero_prod > 0 or zero_yield > 0:
        audit_flags["investigate"].append(f"Zero values present (Production: {zero_prod}, Yield: {zero_yield}). Determine if these represent failed crops or missing data.")

    # Yield derivation check (Yield ~ Production / Area)
    valid_area = df['Area'] > 0
    calculated_yield = df.loc[valid_area, 'Production'] / df.loc[valid_area, 'Area']
    yield_match = np.isclose(df.loc[valid_area, 'Yield'], calculated_yield, rtol=1e-2, atol=1e-2)
    mismatch_count = (~yield_match).sum()
    print(f"\nYield logic check (Yield ≈ Production / Area where Area > 0):")
    print(f"Matches: {yield_match.sum()} | Mismatches: {mismatch_count}")
    if mismatch_count > 0:
        audit_flags["investigate"].append(f"{mismatch_count} rows fail the Yield ≈ Production/Area logic check.")
    else:
        audit_flags["passed"].append("Yield accurately reflects Production / Area.")

    # Quantiles & Top 10
    print("\nYield Quantiles per Crop:")
    quantiles = df.groupby('Crop')['Yield'].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).unstack()
    print(quantiles.to_string())

    print("\nTop 10 Largest Yield Observations per Crop:")
    for crop in crops:
        print(f"\n[{crop}] Top 10:")
        top10 = df[df['Crop'] == crop].nlargest(10, 'Yield')[['District', 'Year', 'Crop', 'Area', 'Production', 'Yield']]
        print(top10.to_string(index=False))

    # -------------------------------------------------------------------------
    # 3. PARTITIONS AND COVERAGE
    # -------------------------------------------------------------------------
    print("\n--- 3. DATA PARTITIONS & COVERAGE ---")
    
    train_mask = df['Year'] <= 2007
    val_mask = (df['Year'] >= 2008) & (df['Year'] <= 2009)
    test_mask = (df['Year'] >= 2010) & (df['Year'] <= 2013)

    print("Partition Boundaries:")
    print("  - Training Period:   Year <= 2007")
    print("  - Validation Period: 2008 <= Year <= 2009")
    print("  - Final Test Period: 2010 <= Year <= 2013")
    print("\nObservation Counts per Crop per Partition:")
    
    counts_df = pd.DataFrame({
        'Train (<=2007)': df[train_mask]['Crop'].value_counts(),
        'Validation (2008-2009)': df[val_mask]['Crop'].value_counts(),
        'Final Test (2010-2013)': df[test_mask]['Crop'].value_counts()
    }).fillna(0).astype(int)
    print(counts_df.to_string())

    # Missing years within District + Crop series
    print("\nAnalyzing missing years within District + Crop timeseries:")
    missing_years_flags = 0
    grouped = df.groupby(['District', 'Crop'])
    for (dist, crop), group in grouped:
        years_present = set(group['Year'])
        if len(years_present) < 2:
            continue
        min_y, max_y = min(years_present), max(years_present)
        expected_years = set(range(min_y, max_y + 1))
        missing = expected_years - years_present
        if missing:
            missing_years_flags += 1
            
    print(f"Number of District+Crop series with internal missing years: {missing_years_flags}")
    if missing_years_flags > 0:
        audit_flags["investigate"].append("Time-series gaps exist within District+Crop histories.")

    # -------------------------------------------------------------------------
    # 4. HISTORICAL LAG AUDIT
    # -------------------------------------------------------------------------
    print("\n--- 4. HISTORICAL LAG FEATURE AUDIT ---")
    print("Recalculating leak-free lag: shift(1).rolling(window=3, min_periods=1).mean()")
    
    df_sorted = df.sort_values(['District', 'Crop', 'Year']).copy()
    
    # Recalculate lag based on rows
    df_sorted['Calculated_Lag'] = df_sorted.groupby(['District', 'Crop'])['Yield'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    
    # Check consecutive calendar years
    df_sorted['Prev_Year_1'] = df_sorted.groupby(['District', 'Crop'])['Year'].shift(1)
    df_sorted['Year_Gap'] = df_sorted['Year'] - df_sorted['Prev_Year_1']
    
    non_consecutive = (df_sorted['Year_Gap'] > 1).sum()
    
    print("\nEXPLANATION OF LAG LOGIC:")
    print("  - shift(1) guarantees strict leak-free behavior by pushing the window backwards.")
    print("  - It prevents the current year's yield from contaminating the current row's prediction.")
    print("  - CAVEAT: row-based rolling averages strictly use the last N rows. If chronological")
    print("    years are missing (e.g., 2004, 2006, 2007), the '3-year window' might span much ")
    print("    older biological conditions (e.g., pulling data from 2003).")
    
    print(f"\nFlags: Found {non_consecutive} instances where the lag calculation utilizes older observations because immediate previous years are missing in the data.")
    
    if 'Yield_Lag_Avg' in df.columns:
        valid_comp = df_sorted.dropna(subset=['Yield_Lag_Avg', 'Calculated_Lag'])
        match = np.isclose(valid_comp['Yield_Lag_Avg'], valid_comp['Calculated_Lag'], equal_nan=True)
        print(f"Comparison with existing 'Yield_Lag_Avg': {match.sum()} matched, {(~match).sum()} mismatched.")
    else:
        print("Existing 'Yield_Lag_Avg' column not found for comparison.")

    # -------------------------------------------------------------------------
    # 5. WEATHER COLUMNS VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- 5. WEATHER DATA VERIFICATION ---")
    weather_keywords = ['Rainfall', 'Temp', 'Humidity', 'Rain', 'T2M', 'RH2M']
    weather_cols = [c for c in df.columns if any(kw.lower() in c.lower() for kw in weather_keywords)]
    
    if weather_cols:
        print(f"Identified weather columns: {weather_cols}")
        print("\nDescriptive Statistics:")
        print(df[weather_cols].describe().T[['count', 'mean', 'std', 'min', 'max']].to_string())
        print("\nMissing Values:")
        print(df[weather_cols].isna().sum().to_string())
        audit_flags["unanswered"].append("Are the weather column units (e.g., mm, Celsius, %) and temporal aggregations mathematically aligned with crop biology? (Requires domain verification)")
    else:
        print("No standard weather columns detected.")
        audit_flags["investigate"].append("Weather features (Rainfall, Temp, Humidity) are missing from the master dataset.")

    # -------------------------------------------------------------------------
    # 6. SOIL DATA QUALITY
    # -------------------------------------------------------------------------
    print("\n--- 6. SOIL DATA VERIFICATION ---")
    if 'Soil_Confidence' in df.columns:
        print("Soil_Confidence categories:")
        conf_counts = df['Soil_Confidence'].value_counts(dropna=False)
        print(conf_counts.to_string())
        districts_per_conf = df.groupby('Soil_Confidence')['District'].nunique()
        print("\nUnique Districts per Confidence Category:")
        print(districts_per_conf.to_string())
    else:
        print("'Soil_Confidence' column not found.")

    soil_nutrients = [c for c in ['pH', 'N', 'P', 'K'] if c in df.columns]
    if soil_nutrients:
        print("\nMissing Values in Soil Parameters:")
        print(df[soil_nutrients].isna().sum().to_string())
    else:
        print("Soil parameters (pH, N, P, K) not found.")

    # -------------------------------------------------------------------------
    # 7. FINAL TEST DIAGNOSTICS (NO PERFORMANCE)
    # -------------------------------------------------------------------------
    print("\n--- 7. FINAL TEST PARTITION DIAGNOSTICS ---")
    df_test = df[test_mask]
    print(f"Total Test Observations (2010-2013): {len(df_test)}")
    print(f"Test Set Year Range: {df_test['Year'].min()} to {df_test['Year'].max()}")
    print("Test Set Missing Values Breakdown:")
    print(df_test.isna().sum().to_string())
    
    # -------------------------------------------------------------------------
    # 8. AUDIT SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" AUDIT SUMMARY & CONCLUSIONS")
    print("=" * 80)
    
    print("\n[✔] PASSED CHECKS:")
    for msg in audit_flags["passed"]:
        print(f"  - {msg}")
        
    print("\n[!] REQUIRES INVESTIGATION:")
    if not audit_flags["investigate"]:
        print("  - None.")
    for msg in audit_flags["investigate"]:
        print(f"  - {msg}")
        
    print("\n[?] UNANSWERED QUESTIONS (Out of Scope for CSV Audit):")
    for msg in audit_flags["unanswered"]:
        print(f"  - {msg}")
    print("  - Do extremely high yields (e.g. Sugarcane) represent correct tonnages or unit scaling errors?")
    print("  - Are zero-production records genuine crop failures, or default placeholder values?")
    print("=" * 80)

if __name__ == "__main__":
    run_validation_audit()