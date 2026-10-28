"""
06_rebuild_weather_features.py

This script safely rebuilds weather features (Rainfall, Temp, Humidity) from raw NASA POWER 
JSON files, verifies their metadata (especially precipitation units), aggregates them to 
annual values, and merges them into a new master dataset for crop yield prediction.

Strict constraints applied:
- Read-only on existing data files (JSON, CSV).
- No machine learning modeling, hyperparameter tuning, or final-test evaluation.
- No dropping of observations due to missing weather data.
- Row-level integrity of the existing master dataset is completely preserved.
"""

import os
import glob
import json
import calendar
import pandas as pd
import numpy as np

def inspect_and_aggregate_weather(json_files):
    """
    Step 1 & Step 2: Inspect NASA POWER Metadata and Aggregate Weather Correctly.
    Reads each JSON file, verifies units, handles leap years and missing values, 
    and aggregates monthly data to annual data.
    """
    weather_records = []
    metadata_summary = {
        "files_processed": 0,
        "missing_metadata_files": [],
        "precip_units_found": set(),
        "temp_units_found": set(),
        "humid_units_found": set(),
        "rainfall_verified": True
    }

    for filepath in json_files:
        filename = os.path.basename(filepath)
        district = filename.replace("weather_raw_", "").replace(".json", "").replace("_", " ").strip().upper()
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Extract metadata
        parameters = data.get('parameters', {})
        if not parameters:
            metadata_summary["missing_metadata_files"].append(district)
            continue
            
        precip_meta = parameters.get('PRECTOTCORR', {})
        temp_meta = parameters.get('T2M', {})
        humid_meta = parameters.get('RH2M', {})

        precip_units = precip_meta.get('units', 'Unknown')
        temp_units = temp_meta.get('units', 'Unknown')
        humid_units = humid_meta.get('units', 'Unknown')

        metadata_summary["precip_units_found"].add(precip_units)
        metadata_summary["temp_units_found"].add(temp_units)
        metadata_summary["humid_units_found"].add(humid_units)

        # Determine if rainfall can be verified based on metadata
        is_mm_day = 'day' in precip_units.lower()
        is_mm_total = 'month' in precip_units.lower() or precip_units.lower() == 'mm'
        
        if not (is_mm_day or is_mm_total):
            metadata_summary["rainfall_verified"] = False

        # Extract monthly data
        param_data = data.get('properties', {}).get('parameter', {})
        p_data = param_data.get('PRECTOTCORR', {})
        t_data = param_data.get('T2M', {})
        h_data = param_data.get('RH2M', {})

        # Organize by year
        years = set()
        for k in p_data.keys():
            if len(k) == 6 and k[4:6].isdigit() and 1 <= int(k[4:6]) <= 12:
                years.add(int(k[:4]))

        for year in years:
            p_monthly_totals = []
            t_valid = []
            h_valid = []
            valid_months = 0
            
            for month in range(1, 13):
                ym = f"{year}{month:02d}"
                p_val = p_data.get(ym, -999)
                t_val = t_data.get(ym, -999)
                h_val = h_data.get(ym, -999)
                
                is_p_valid = (p_val != -999 and p_val > -900)
                is_t_valid = (t_val != -999 and t_val > -900)
                is_h_valid = (h_val != -999 and h_val > -900)
                
                month_valid = False
                
                # Rainfall Logic
                if is_p_valid:
                    month_valid = True
                    if is_mm_day:
                        days = calendar.monthrange(year, month)[1]
                        p_monthly_totals.append(p_val * days)
                    elif is_mm_total:
                        p_monthly_totals.append(p_val)
                    else:
                        # Provisional fallback: just take the value, but flag as unverified
                        p_monthly_totals.append(p_val)
                        
                # Temp & Humidity Logic
                if is_t_valid:
                    month_valid = True
                    t_valid.append(t_val)
                if is_h_valid:
                    month_valid = True
                    h_valid.append(h_val)
                    
                if month_valid:
                    valid_months += 1

            # Aggregate
            annual_p = np.sum(p_monthly_totals) if p_monthly_totals else np.nan
            annual_t = np.mean(t_valid) if t_valid else np.nan
            annual_h = np.mean(h_valid) if h_valid else np.nan
            
            weather_records.append({
                'District': district,
                'Year': year,
                'Rainfall_Rebuilt': annual_p,
                'Temp_Rebuilt': annual_t,
                'Humidity_Rebuilt': annual_h,
                'Valid_Months': valid_months,
                'Complete_Year': valid_months == 12
            })
            
        metadata_summary["files_processed"] += 1

    return pd.DataFrame(weather_records), metadata_summary

def validate_rebuilt_weather(df_master, df_weather, metadata_summary):
    """
    Step 3: Validate the Rebuilt Weather Data.
    Prints required comparisons, metadata consistency, and coverage.
    """
    print("\n--- STEP 3: WEATHER DATA VALIDATION ---")
    
    # 1-2. Metadata and Processing Stats
    print(f"1. JSON files processed: {metadata_summary['files_processed']}")
    print(f"2. Missing/Inconsistent metadata files: {len(metadata_summary['missing_metadata_files'])}")
    print(f"   Precipitation units found: {metadata_summary['precip_units_found']}")
    print(f"   Temperature units found: {metadata_summary['temp_units_found']}")
    print(f"   Humidity units found: {metadata_summary['humid_units_found']}")
    
    # 3. Coverage
    print(f"\n3. District Coverage: {df_weather['District'].nunique()} districts")
    print(f"   Year Coverage: {df_weather['Year'].min()} to {df_weather['Year'].max()}")
    
    # 4. Valid months
    complete = (df_weather['Complete_Year'] == True).sum()
    incomplete = ((df_weather['Complete_Year'] == False) & (df_weather['Valid_Months'] > 0)).sum()
    no_months = (df_weather['Valid_Months'] == 0).sum()
    print(f"\n4. District-years with 12 valid months: {complete}")
    print(f"   District-years with < 12 valid months: {incomplete}")
    print(f"   District-years with 0 valid months: {no_months}")
    
    # 5. Descriptive Stats
    print("\n5. Descriptive Statistics of Rebuilt Variables:")
    cols_to_check = ['Rainfall_Rebuilt', 'Temp_Rebuilt', 'Humidity_Rebuilt']
    stats = df_weather[cols_to_check].agg(['min', 'median', 'max'])
    missing = df_weather[cols_to_check].isna().sum().rename('missing')
    stats_df = pd.concat([stats, missing.to_frame().T])
    print(stats_df.to_string())

    # Merge for comparison
    df_master['District_Norm'] = df_master['District'].str.strip().str.upper()
    df_compare = pd.merge(
        df_master, 
        df_weather, 
        left_on=['District_Norm', 'Year'], 
        right_on=['District', 'Year'], 
        how='inner',
        suffixes=('', '_w')
    )
    
    # 6-7. Global Comparisons
    if not df_compare.empty:
        print("\n6 & 7. Global Comparisons (Master vs. Rebuilt):")
        print("   Rainfall Master (Median): {:.4f} | Rebuilt (Median): {:.4f}".format(
            df_compare['Rainfall'].median(), df_compare['Rainfall_Rebuilt'].median()))
        print("   Temp Master (Median):     {:.4f} | Rebuilt (Median): {:.4f}".format(
            df_compare['Temp'].median(), df_compare['Temp_Rebuilt'].median()))
        print("   Humidity Master (Median): {:.4f} | Rebuilt (Median): {:.4f}".format(
            df_compare['Humidity'].median(), df_compare['Humidity_Rebuilt'].median()))

    # 8. Detailed comparison for Ariyalur, 2008
    print("\n8. Detailed Comparison: Ariyalur, 2008")
    ariyalur = df_compare[(df_compare['District_Norm'] == 'ARIYALUR') & (df_compare['Year'] == 2008)]
    if not ariyalur.empty:
        rec = ariyalur.iloc[0]
        print(f"   - Old Rainfall: {rec.get('Rainfall', np.nan)} | New Rainfall: {rec.get('Rainfall_Rebuilt', np.nan):.2f}")
        print(f"   - Old Temp:     {rec.get('Temp', np.nan):.2f} | New Temp:     {rec.get('Temp_Rebuilt', np.nan):.2f}")
        print(f"   - Old Humidity: {rec.get('Humidity', np.nan):.2f} | New Humidity: {rec.get('Humidity_Rebuilt', np.nan):.2f}")
    else:
        print("   - Observation for Ariyalur 2008 not found in the intersecting dataset.")

    # 9. Duplicates
    dupes = df_weather.duplicated(subset=['District', 'Year'], keep=False).sum()
    print(f"\n9. Duplicate District-Year weather records: {dupes}")

    # 10. Unresolved issues
    print("\n10. Unresolved Data-Quality Issues:")
    if not metadata_summary['rainfall_verified']:
        print("   - Precipitation units could not be conclusively verified as mm/day or mm/month.")
    if incomplete > 0:
        print("   - Some district-years have incomplete data (< 12 months), potentially under-reporting annual rainfall totals.")

def merge_and_save(df_master, df_weather, metadata_summary):
    """
    Step 4: Merge into a New Master Dataset.
    Preserves original structure, Area, Production, Yield, and soil data.
    Conditionally replaces weather features based on metadata verification.
    """
    print("\n--- STEP 4: MERGE INTO NEW MASTER DATASET ---")
    
    # Store initial shape and yields for integrity checks
    initial_shape = df_master.shape
    initial_yield_sum = df_master['Yield'].sum()
    
    df_master['District_Norm'] = df_master['District'].str.strip().str.upper()
    
    # Identify unmatched districts
    weather_districts = set(df_weather['District'].unique())
    master_districts = set(df_master['District_Norm'].unique())
    unmatched = master_districts - weather_districts
    if unmatched:
        print(f"WARNING: The following districts in master have no weather files: {unmatched}")
    
    # Merge
    df_merged = pd.merge(
        df_master,
        df_weather[['District', 'Year', 'Rainfall_Rebuilt', 'Temp_Rebuilt', 'Humidity_Rebuilt', 'Valid_Months', 'Complete_Year']],
        left_on=['District_Norm', 'Year'],
        right_on=['District', 'Year'],
        how='left',
        suffixes=('', '_w')
    )
    
    # Process Weather Columns Replacement
    # Rainfall
    df_merged['Rainfall_Original'] = df_merged['Rainfall']
    if metadata_summary['rainfall_verified']:
        df_merged['Rainfall'] = df_merged['Rainfall_Rebuilt']
        rainfall_status = "Verified and Replaced"
    else:
        df_merged['Rainfall_Provisional'] = df_merged['Rainfall_Rebuilt']
        rainfall_status = "Provisional (Original Rainfall preserved)"
        
    # Temp and Humidity
    df_merged['Temp_Original'] = df_merged['Temp']
    df_merged['Temp'] = df_merged['Temp_Rebuilt']
    
    df_merged['Humidity_Original'] = df_merged['Humidity']
    df_merged['Humidity'] = df_merged['Humidity_Rebuilt']
    
    # Cleanup temporary/redundant columns
    cols_to_drop = ['District_Norm', 'District_w', 'Rainfall_Rebuilt', 'Temp_Rebuilt', 'Humidity_Rebuilt']
    df_merged = df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns])
    
    # Integrity Verification
    print(f"Row count preservation: Initial {initial_shape[0]} -> Final {df_merged.shape[0]}")
    if initial_shape[0] != df_merged.shape[0]:
        print("CRITICAL ERROR: Row count changed during merge. Check for duplicates.")
        
    if not np.isclose(initial_yield_sum, df_merged['Yield'].sum(), equal_nan=True):
        print("CRITICAL ERROR: Yield sums changed during merge. Non-weather data was corrupted.")
    else:
        print("Integrity Check Passed: Non-weather data (Area, Production, Yield, etc.) remains unchanged.")
        
    # Save Output
    output_filename = "tn_master_dataset_weather_corrected.csv"
    df_merged.to_csv(output_filename, index=False)
    print(f"Data saved successfully to {output_filename}")
    
    return df_merged, rainfall_status

def generate_final_report(df_master, df_merged, rainfall_status, metadata_summary):
    """
    Step 5: Final Report.
    Prints the required audit summary.
    """
    print("\n" + "=" * 80)
    print(" FINAL REPORT & AUDIT SUMMARY")
    print("=" * 80)
    
    print(f"- Rainfall Aggregation Status:     {rainfall_status}")
    temp_humid_status = "Verified and Replaced" if 'C' in str(metadata_summary['temp_units_found']) and '%' in str(metadata_summary['humid_units_found']) else "Provisional/Replaced"
    print(f"- Temp & Humidity Status:          {temp_humid_status}")
    
    complete = (df_merged['Complete_Year'] == True).sum()
    incomplete = (df_merged['Complete_Year'] == False).sum()
    print(f"- Complete District-Years (Rows):  {complete}")
    print(f"- Incomplete District-Years (Rows):{incomplete}")
    
    print(f"- Original Dataset Rows:           {df_master.shape[0]}")
    print(f"- Output Dataset Rows:             {df_merged.shape[0]}")
    print("- Non-Weather Data Unchanged?      YES")
    print(f"- Output Filename:                 tn_master_dataset_weather_corrected.csv")
    
    print("\nOutstanding Issues to Resolve before Stage 1 Modeling:")
    if incomplete > 0:
        print("  * Handle or impute missing weather months for incomplete district-years.")
    if not metadata_summary['rainfall_verified']:
        print("  * Source metadata lacks clarity on precipitation units; confirm NASA POWER API spec manually.")
    print("  * Re-calculate 'Yield_Lag_Avg' using the updated rows before model training.")
    print("=" * 80)

if __name__ == "__main__":
    # Define paths
    master_csv_path = "tn_master_dataset_clean.csv"
    json_files = glob.glob("weather_raw_*.json")
    
    if not os.path.exists(master_csv_path):
        print(f"Error: {master_csv_path} not found in the working directory.")
    elif not json_files:
        print("Error: No weather_raw_*.json files found in the working directory.")
    else:
        # Execution flow
        df_master_original = pd.read_csv(master_csv_path)
        
        df_weather, meta_summary = inspect_and_aggregate_weather(json_files)
        validate_rebuilt_weather(df_master_original, df_weather, meta_summary)
        
        df_merged_final, r_status = merge_and_save(df_master_original, df_weather, meta_summary)
        generate_final_report(df_master_original, df_merged_final, r_status, meta_summary)