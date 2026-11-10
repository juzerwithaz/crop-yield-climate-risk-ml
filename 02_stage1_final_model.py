import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def run_lag_pipeline():
    print("Loading tn_master_dataset_clean.csv...")
    df = pd.read_csv("tn_master_dataset_clean.csv")

    # 1. Sort the data rigorously to ensure chronological order for rolling averages
    df = df.sort_values(by=['District', 'Crop', 'Year']).reset_index(drop=True)

    # 2. Create the Leak-Free Historical Trend Feature
    # shift(1) guarantees we only look at past years. 
    # rolling(window=3, min_periods=1) averages up to 3 previous years.
    df['Yield_Lag_Avg'] = df.groupby(['District', 'Crop'])['Yield'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # 3. Drop rows with NaN lag (the very first recorded year for every district+crop)
    initial_len = len(df)
    df = df.dropna(subset=['Yield_Lag_Avg']).copy()
    print(f"Dropped {initial_len - len(df)} rows (the first year of data for each District+Crop) to maintain a strict, leak-free historical lag.\n")

    # Log transform target
    df['Log_Yield'] = np.log1p(df['Yield'])

    # Feature sets
    features_climate_only = ['Area', 'Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K']
    features_with_lag = features_climate_only + ['Yield_Lag_Avg']
    target = 'Yield'

    crops = sorted(df['Crop'].unique())
    summary_results = []

    print("="*65)
    print(" TRAINING MODELS (Year <= 2009) & TESTING (Year >= 2010)")
    print("="*65)

    for crop in crops:
        crop_df = df[df['Crop'] == crop]
        
        train_df = crop_df[crop_df['Year'] <= 2009]
        test_df = crop_df[crop_df['Year'] >= 2010]
        
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Skipping {crop} due to insufficient split data.")
            continue

        y_train_log = train_df['Log_Yield']
        y_test_orig = test_df[target]

        # --- MODEL A: Climate Only (For comparison) ---
        rf_base = RandomForestRegressor(random_state=42, n_estimators=100, min_samples_leaf=3, max_depth=12)
        rf_base.fit(train_df[features_climate_only], y_train_log)
        preds_base = np.expm1(rf_base.predict(test_df[features_climate_only]))
        r2_base = r2_score(y_test_orig, preds_base)

        # --- MODEL B: Climate + Yield Lag (The New Model) ---
        rf_lag = RandomForestRegressor(random_state=42, n_estimators=100, min_samples_leaf=3, max_depth=12)
        rf_lag.fit(train_df[features_with_lag], y_train_log)
        preds_lag = np.expm1(rf_lag.predict(test_df[features_with_lag]))

        rmse = root_mean_squared_error(y_test_orig, preds_lag)
        mae = mean_absolute_error(y_test_orig, preds_lag)
        r2_lag = r2_score(y_test_orig, preds_lag)

        # Extract Importances
        importances = pd.DataFrame({
            'Feature': features_with_lag,
            'Importance': rf_lag.feature_importances_ * 100
        }).sort_values(by='Importance', ascending=False)

        summary_results.append({
            'Crop': crop,
            'Usable Rows': len(crop_df),
            'Old R2': r2_base,
            'New R2': r2_lag
        })

        print(f"\n--- {crop.upper()} ---")
        print(f"Usable Rows: {len(crop_df)} (Train: {len(train_df)} | Test: {len(test_df)})")
        print(f"Metrics (New): RMSE = {rmse:.2f} | MAE = {mae:.2f} | R² = {r2_lag:.4f}")
        print("Top 5 Features:")
        for _, row in importances.head(5).iterrows():
            print(f"  - {row['Feature']:<14}: {row['Importance']:.1f}%")

    # --- FINAL COMPARISON TABLE ---
    print("\n" + "="*70)
    print(" R² SUMMARY COMPARISON (Test Set: 2010-2013)")
    print("="*70)
    print(f"{'Crop':<15} | {'Climate-Only R²':<17} | {'Climate + Lag R²'}")
    print("-" * 70)
    for res in summary_results:
        print(f"{res['Crop']:<15} | {res['Old R2']:>17.4f} | {res['New R2']:>15.4f}")
    print("="*70)

if __name__ == "__main__":
    run_lag_pipeline()