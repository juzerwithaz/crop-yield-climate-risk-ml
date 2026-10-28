import pandas as pd
import numpy as np
import glob
import os
import json

def build_master_dataset():
    # --- 1. Load and process yield data ---
    print("Loading Yield Data...")
    df_yield = pd.read_csv("tn_crop_yield_filtered.csv")

    # Ensure standard column names
    df_yield.rename(columns={'District_Name': 'District', 'Crop_Year': 'Year'}, inplace=True)
    df_yield['District'] = df_yield['District'].astype(str).str.strip().str.upper()

    # Aggregate to ensure exactly ONE row per District + Year + Crop
    # (This naturally handles multiple seasons by summing total Area and Production)
    df_yield = df_yield.groupby(['District', 'Year', 'Crop'], as_index=False)[['Area', 'Production']].sum()

    # Compute Yield (Production / Area), avoiding division by zero
    df_yield['Yield'] = np.where(df_yield['Area'] > 0, df_yield['Production'] / df_yield['Area'], np.nan)


    # --- 2. Load and aggregate weather data ---
    print("Loading and Aggregating Weather Data...")
    weather_files = glob.glob("weather_raw_*.json")
    weather_records = []

    for file in weather_files:
        # Extract district from filename: weather_raw_the_nilgiris.json -> THE NILGIRIS
        base_name = os.path.basename(file).replace("weather_raw_", "").replace(".json", "")
        district = base_name.replace("_", " ").strip().upper()
        
        with open(file, 'r') as f:
            data = json.load(f)
            
        params = data['properties']['parameter']
        
        # Iterate through the time keys (YYYYMM format)
        for ym in params['PRECTOTCORR'].keys():
            if len(ym) == 6:
                year = int(ym[:4])
                month = int(ym[4:])
                
                # Average across the 12 months (ignores '13' which NASA sometimes outputs for annual totals)
                if 1 <= month <= 12: 
                    precip = params['PRECTOTCORR'].get(ym, np.nan)
                    temp = params['T2M'].get(ym, np.nan)
                    humidity = params['RH2M'].get(ym, np.nan)
                    
                    weather_records.append({
                        'District': district,
                        'Year': year,
                        'Rainfall': precip if precip > -900 else np.nan, # NASA missing value is -999
                        'Temp': temp if temp > -900 else np.nan,
                        'Humidity': humidity if humidity > -900 else np.nan
                    })

    df_weather_monthly = pd.DataFrame(weather_records)

    # Aggregate to YEARLY average per district
    df_weather_yearly = df_weather_monthly.groupby(['District', 'Year'], as_index=False).mean()


    # --- 3. Load Soil Data ---
    print("Loading Soil Data...")
    # Supports both your local file name and the requested file name
    soil_file = "tn_soil_data.csv" if os.path.exists("tn_soil_data.csv") else "TN_Soil_Data_31_Districts.csv"
    df_soil = pd.read_csv(soil_file)

    # Normalize column names for a clean merge
    rename_map = {}
    for col in df_soil.columns:
        if 'N (' in col or col == 'N': rename_map[col] = 'N'
        elif 'P (' in col or col == 'P': rename_map[col] = 'P'
        elif 'K (' in col or col == 'K': rename_map[col] = 'K'
        elif 'Confidence' in col: rename_map[col] = 'Soil_Confidence'

    df_soil.rename(columns=rename_map, inplace=True)
    df_soil['District'] = df_soil['District'].astype(str).str.strip().str.upper()


    # --- 4. Merge Data ---
    print("Merging Datasets...")
    # Join yield with weather
    df_master = pd.merge(df_yield, df_weather_yearly, on=['District', 'Year'], how='left')
    # Join with static soil data
    df_master = pd.merge(df_master, df_soil[['District', 'pH', 'N', 'P', 'K', 'Soil_Confidence']], on='District', how='left')

    # Enforce exact final column layout requested
    final_cols = ['District', 'Year', 'Crop', 'Area', 'Production', 'Yield', 
                  'Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K', 'Soil_Confidence']
    df_master = df_master.reindex(columns=final_cols)


    # --- 5. Diagnostics & Output (Raw) ---
    print("\n=== DATASET DIAGNOSTICS ===")
    total_rows = len(df_master)
    print(f"(a) Total Row Count: {total_rows}")

    rows_with_nulls = df_master.isnull().any(axis=1).sum()
    print(f"(b) Rows with ANY Null Values: {rows_with_nulls}")

    print("(c) Breakdown of Nulls by Column:")
    null_counts = df_master.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"    - {col}: {count} missing values")

    print("(d) Missing Districts Report:")
    yield_dists = set(df_yield['District'].unique())
    weather_dists = set(df_weather_yearly['District'].unique())
    soil_dists = set(df_soil['District'].unique())

    missing_in_weather = yield_dists - weather_dists
    missing_in_soil = yield_dists - soil_dists

    if not missing_in_weather and not missing_in_soil:
        print("    - OK: All districts in yield data are successfully matched in weather and soil data.")
    else:
        if missing_in_weather:
            print(f"    - WARNING: Districts in Yield but MISSING in Weather: {missing_in_weather}")
        if missing_in_soil:
            print(f"    - WARNING: Districts in Yield but MISSING in Soil: {missing_in_soil}")

    # Save RAW dataset
    df_master.to_csv("tn_master_dataset_raw.csv", index=False)
    print("\nSaved full uncleaned dataset to -> tn_master_dataset_raw.csv")


    # --- 6. Cleaning Step ---
    # Drop rows missing Yield, Rainfall, or Temp (implicitly keeping gaps that are ONLY in soil data)
    df_clean = df_master.dropna(subset=['Yield', 'Rainfall', 'Temp'])

    print("\n=== CLEANING STEP ===")
    print(f"Rows Before Cleaning: {total_rows}")
    print(f"Rows After Cleaning:  {len(df_clean)}")
    print(f"Total Rows Dropped:   {total_rows - len(df_clean)}")

    df_clean.to_csv("tn_master_dataset_clean.csv", index=False)
    print("Saved cleaned dataset to -> tn_master_dataset_clean.csv")

if __name__ == "__main__":
    build_master_dataset()