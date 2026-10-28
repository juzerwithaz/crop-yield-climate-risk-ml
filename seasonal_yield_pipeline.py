import os
import glob
import json
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings from empty slices if some months are entirely missing
warnings.filterwarnings("ignore", category=RuntimeWarning)

def build_and_evaluate_seasonal_model():
    print("=== STEP 1: Rebuilding Weather Aggregation (Growing Season) ===")
    weather_files = glob.glob("weather_raw_*.json")
    weather_records = []

    for file in weather_files:
        base_name = os.path.basename(file).replace("weather_raw_", "").replace(".json", "")
        district = base_name.replace("_", " ").strip().upper()
        
        with open(file, 'r') as f:
            data = json.load(f)
            
        params = data['properties']['parameter']
        precip_data = params.get('PRECTOTCORR', {})
        temp_data = params.get('T2M', {})
        humid_data = params.get('RH2M', {})
        
        # Organize data by year and month
        year_dict = {}
        for ym in precip_data.keys():
            if len(ym) == 6:
                year = int(ym[:4])
                month = int(ym[4:])
                if 1 <= month <= 12: # Ignore annual summaries (e.g., month '13')
                    if year not in year_dict:
                        year_dict[year] = {'P': {}, 'T': {}, 'H': {}}
                    
                    p_val = precip_data.get(ym, -999)
                    t_val = temp_data.get(ym, -999)
                    h_val = humid_data.get(ym, -999)
                    
                    year_dict[year]['P'][month] = p_val if p_val > -900 else np.nan
                    year_dict[year]['T'][month] = t_val if t_val > -900 else np.nan
                    year_dict[year]['H'][month] = h_val if h_val > -900 else np.nan

        # Compute seasonal metrics per year
        for year, monthly in year_dict.items():
            gs_months = [6, 7, 8, 9]  # June to September
            all_months = list(range(1, 13))
            
            # Extract valid values (ignoring NaNs)
            gs_p = [monthly['P'].get(m) for m in gs_months if not np.isnan(monthly['P'].get(m, np.nan))]
            gs_t = [monthly['T'].get(m) for m in gs_months if not np.isnan(monthly['T'].get(m, np.nan))]
            gs_h = [monthly['H'].get(m) for m in gs_months if not np.isnan(monthly['H'].get(m, np.nan))]
            ann_p = [monthly['P'].get(m) for m in all_months if not np.isnan(monthly['P'].get(m, np.nan))]
            
            weather_records.append({
                'District': district,
                'Year': year,
                'Growing_Season_Rainfall': sum(gs_p) if gs_p else np.nan,
                'Growing_Season_Temp': sum(gs_t)/len(gs_t) if gs_t else np.nan,
                'Growing_Season_Humidity': sum(gs_h)/len(gs_h) if gs_h else np.nan,
                'Annual_Rainfall_Total': sum(ann_p) if ann_p else np.nan
            })

    df_weather = pd.DataFrame(weather_records)

    print("=== STEP 2: Rebuilding Master Dataset ===")
    df_yield = pd.read_csv("tn_crop_yield_filtered.csv")
    df_yield.rename(columns={'District_Name': 'District', 'Crop_Year': 'Year'}, inplace=True)
    df_yield['District'] = df_yield['District'].astype(str).str.strip().str.upper()

    # Aggregate Area and Production
    df_yield = df_yield.groupby(['District', 'Year', 'Crop'], as_index=False)[['Area', 'Production']].sum()
    df_yield['Yield'] = np.where(df_yield['Area'] > 0, df_yield['Production'] / df_yield['Area'], np.nan)

    # Load Soil Data
    soil_file = "tn_soil_data.csv" if os.path.exists("tn_soil_data.csv") else "TN_Soil_Data_31_Districts.csv"
    df_soil = pd.read_csv(soil_file)
    rename_map = {}
    for col in df_soil.columns:
        if 'N (' in col or col == 'N': rename_map[col] = 'N'
        elif 'P (' in col or col == 'P': rename_map[col] = 'P'
        elif 'K (' in col or col == 'K': rename_map[col] = 'K'
    df_soil.rename(columns=rename_map, inplace=True)
    df_soil['District'] = df_soil['District'].astype(str).str.strip().str.upper()

    # Merge
    df_master = pd.merge(df_yield, df_weather, on=['District', 'Year'], how='left')
    df_master = pd.merge(df_master, df_soil[['District', 'pH', 'N', 'P', 'K']], on='District', how='left')
    
    # Save v2
    df_master.to_csv("tn_master_dataset_v2.csv", index=False)
    print("Saved -> tn_master_dataset_v2.csv")

    print("=== STEP 3: Rebuilding Yield_Lag_Avg ===")
    df_master = df_master.sort_values(by=['District', 'Crop', 'Year']).reset_index(drop=True)
    df_master['Yield_Lag_Avg'] = df_master.groupby(['District', 'Crop'])['Yield'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df_master = df_master.dropna(subset=['Yield_Lag_Avg', 'Growing_Season_Temp']).copy()
    
    # Target transform
    df_master['Log_Yield'] = np.log1p(df_master['Yield'])

    print("=== STEP 4: Retraining & Comparing Models ===")
    features = [
        'Area', 'Growing_Season_Rainfall', 'Growing_Season_Temp', 
        'Growing_Season_Humidity', 'Annual_Rainfall_Total', 
        'pH', 'N', 'P', 'K', 'Yield_Lag_Avg'
    ]
    target = 'Yield'
    crops = sorted(df_master['Crop'].unique())

    previous_r2 = {
        'Cotton(lint)': -0.21,
        'Groundnut': 0.21,
        'Maize': -0.05,
        'Rice': 0.07,
        'Sugarcane': 0.12
    }
    
    results = []

    for crop in crops:
        crop_df = df_master[df_master['Crop'] == crop]
        train_df = crop_df[crop_df['Year'] <= 2009]
        test_df = crop_df[crop_df['Year'] >= 2010]
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
            
        X_train = train_df[features]
        y_train_log = train_df['Log_Yield']
        X_test = test_df[features]
        y_test_orig = test_df[target]
        
        rf = RandomForestRegressor(random_state=42, n_estimators=100, min_samples_leaf=3, max_depth=12)
        rf.fit(X_train, y_train_log)
        
        y_pred = np.expm1(rf.predict(X_test))
        rmse = mean_squared_error(y_test_orig, y_pred) ** 0.5
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        
        # Feature importances
        importances = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_ * 100
        }).sort_values(by='Importance', ascending=False).head(5)
        
        results.append({
            'Crop': crop,
            'Old_R2': previous_r2.get(crop, np.nan),
            'New_R2': r2
        })
        
        print(f"\n--- {crop.upper()} ---")
        print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
        print("Top 5 Features:")
        for _, row in importances.iterrows():
            print(f"  - {row['Feature']:<25}: {row['Importance']:.1f}%")

    print("\n" + "="*65)
    print(" R² SUMMARY COMPARISON: Full-Year vs Growing-Season")
    print("="*65)
    print(f"{'Crop':<15} | {'Previous R² (Full)':<20} | {'New R² (Seasonal)'}")
    print("-" * 65)
    for res in results:
        print(f"{res['Crop']:<15} | {res['Old_R2']:>18.2f} | {res['New_R2']:>17.4f}")
    print("="*65)

if __name__ == "__main__":
    build_and_evaluate_seasonal_model()