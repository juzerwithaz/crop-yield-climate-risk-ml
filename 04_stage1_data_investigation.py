import pandas as pd
import json
import glob
import os

def run_data_investigation():
    print("=" * 80)
    print(" 04_STAGE1_DATA_INVESTIGATION: ANOMALY & METADATA AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. SUGARCANE EXTREME YIELD INVESTIGATION
    # -------------------------------------------------------------------------
    print("\n--- 1. SUGARCANE EXTREME YIELD COMPARISON ---")
    raw_file = "tn_crop_yield_filtered.csv"
    clean_file = "tn_master_dataset_clean.csv"

    if os.path.exists(raw_file) and os.path.exists(clean_file):
        df_raw = pd.read_csv(raw_file)
        df_clean = pd.read_csv(clean_file)

        # Standardize raw dataset columns for comparison
        if 'District_Name' in df_raw.columns:
            df_raw.rename(columns={'District_Name': 'District', 'Crop_Year': 'Year'}, inplace=True)
        
        # Calculate yield on the raw dataset
        valid_area_raw = df_raw['Area'] > 0
        df_raw.loc[valid_area_raw, 'Raw_Calculated_Yield'] = df_raw.loc[valid_area_raw, 'Production'] / df_raw.loc[valid_area_raw, 'Area']

        # Filter for Sugarcane
        sugar_raw = df_raw[df_raw['Crop'].str.contains('Sugarcane', case=False, na=False)].copy()
        sugar_clean = df_clean[df_clean['Crop'].str.contains('Sugarcane', case=False, na=False)].copy()

        print("Top 10 Sugarcane Yields in ORIGINAL Data (tn_crop_yield_filtered.csv):")
        top10_raw = sugar_raw.nlargest(10, 'Raw_Calculated_Yield')
        print(top10_raw[['District', 'Year', 'Crop', 'Area', 'Production', 'Raw_Calculated_Yield']].to_string(index=False))

        print("\nTop 10 Sugarcane Yields in MASTER Data (tn_master_dataset_clean.csv):")
        top10_clean = sugar_clean.nlargest(10, 'Yield')
        print(top10_clean[['District', 'Year', 'Crop', 'Area', 'Production', 'Yield']].to_string(index=False))
        
        print("\nINSIGHT: If these extreme yields (>1000 tonnes/ha) exist in the raw data, the issue ")
        print("is source-level unit scaling (e.g., Production reported in kgs or raw cane stalks instead of tonnes) ")
        print("rather than a pipeline aggregation bug.")
    else:
        print(f"Missing files for yield comparison. Need {raw_file} and {clean_file}.")

    # -------------------------------------------------------------------------
    # 2. NASA POWER WEATHER METADATA & UNIT AUDIT
    # -------------------------------------------------------------------------
    print("\n--- 2. NASA POWER WEATHER METADATA AUDIT ---")
    weather_files = glob.glob("weather_raw_*.json")
    
    if weather_files:
        sample_file = weather_files[0]
        print(f"Inspecting sample file: {sample_file}")
        
        with open(sample_file, 'r') as f:
            data = json.load(f)
            
        # Extract headers and parameters
        parameters = data.get('parameters', {})
        header = data.get('header', {})
        
        print("\nParameter Definitions & Units:")
        for param, details in parameters.items():
            units = details.get('units', 'Unknown')
            longname = details.get('longname', 'Unknown')
            print(f"  - {param}: {longname} | Units: {units}")
            
        print("\nCRITICAL RAINFALL AGGREGATION CHECK:")
        precip_units = parameters.get('PRECTOTCORR', {}).get('units', '').lower()
        print(f"Rainfall unit detected as: '{precip_units}'")
        
        if 'day' in precip_units:
            print("WARNING: Precipitation is provided as a DAILY AVERAGE (mm/day) for the month.")
            print("Simply summing these values for 4 months yields a meaningless number (~40-80mm).")
            print("To get total seasonal rainfall, you MUST multiply each monthly value by the number of days in that month before summing.")
        elif 'month' in precip_units or precip_units == 'mm':
            print("Precipitation is provided as TOTAL ACCUMULATED mm per month. Direct summation is mathematically valid.")
        
        # Extract a sample year (e.g., 2005) to observe the data structure
        print("\nSample Monthly Values for Year 2005:")
        param_data = data.get('properties', {}).get('parameter', {})
        precip_data = param_data.get('PRECTOTCORR', {})
        temp_data = param_data.get('T2M', {})
        humid_data = param_data.get('RH2M', {})
        
        sample_months = [f"2005{str(m).zfill(2)}" for m in range(1, 13)]
        for ym in sample_months:
            p = precip_data.get(ym, 'Missing')
            t = temp_data.get(ym, 'Missing')
            h = humid_data.get(ym, 'Missing')
            print(f"  {ym} -> PRECTOTCORR: {p}, T2M: {t}, RH2M: {h}")

    else:
        print("No weather_raw_*.json files found in the current directory.")

    # -------------------------------------------------------------------------
    # 3. ZERO-PRODUCTION ROWS AUDIT
    # -------------------------------------------------------------------------
    print("\n--- 3. ZERO-PRODUCTION ROWS AUDIT ---")
    if os.path.exists(clean_file):
        df_clean = pd.read_csv(clean_file)
        zero_mask = (df_clean['Production'] == 0) | (df_clean['Yield'] == 0)
        zeros_df = df_clean[zero_mask]
        
        print(f"Found {len(zeros_df)} rows with Zero Production or Zero Yield.")
        
        if len(zeros_df) > 0:
            print("\nBreakdown of Zero-Production rows by Crop:")
            print(zeros_df['Crop'].value_counts().to_string())
            
            print("\nDetailed List of Zero-Production Rows:")
            print(zeros_df[['District', 'Year', 'Crop', 'Area', 'Production', 'Yield']].to_string(index=False))
            
            print("\nINSIGHT: Check if 'Area' is greater than 0 for these rows. If Area > 0 but Production = 0, ")
            print("this represents a total crop failure (e.g., severe drought/flood). If Area is also 0, it simply ")
            print("means the crop was not planted in that district that year.")
    else:
        print(f"Missing {clean_file} for zero-production audit.")

    print("\n" + "=" * 80)
    print(" END OF INVESTIGATION")
    print("=" * 80)

if __name__ == "__main__":
    run_data_investigation()