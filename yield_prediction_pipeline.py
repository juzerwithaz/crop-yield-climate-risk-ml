import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def run_honest_pipeline():
    print("Loading data...")
    df = pd.read_csv("tn_master_dataset_clean.csv")

    # Log transform to handle any internal scale skew within a specific crop
    df['Log_Yield'] = np.log1p(df['Yield'])

    # We keep 'Year' so the model can track the agricultural technology trend, 
    # leaving Weather and Soil to explain the climate risk/variance.
    features = ['Year', 'Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K', 'Area']
    
    unique_crops = sorted(df['Crop'].unique())
    
    print("\n=======================================================")
    print("  HONEST PIPELINE: INDIVIDUAL CLIMATE RISK BY CROP")
    print("=======================================================\n")

    for crop in unique_crops:
        crop_df = df[df['Crop'] == crop].copy()
        
        X = crop_df[features]
        y_log = crop_df['Log_Yield']
        y_orig = crop_df['Yield']
        
        # Standard Random Split (80/20) to fairly evaluate weather patterns 
        # across the entire historical dataset.
        X_train, X_test, y_train_log, y_test_log, _, y_test_orig = train_test_split(
            X, y_log, y_orig, test_size=0.20, random_state=42
        )
        
        # Train isolated model for this specific crop
        rf = RandomForestRegressor(
            random_state=42, 
            n_estimators=100, 
            min_samples_leaf=3, 
            max_depth=12
        )
        rf.fit(X_train, y_train_log)
        
        # Predict and inverse transform
        y_pred_log = rf.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        
        # Metrics
        rmse = root_mean_squared_error(y_test_orig, y_pred)
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        
        print(f"--- {crop.upper()} ---")
        print(f"Data points: {len(crop_df)} (Train: {len(X_train)} | Test: {len(X_test)})")
        print(f"RMSE: {rmse:.2f} tonnes/ha")
        print(f"MAE:  {mae:.2f} tonnes/ha")
        print(f"R²:   {r2:.4f}\n")
        
        # (Optional) Print the top 3 features driving this specific crop's yield
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("Top 3 yield drivers:")
        for feat, imp in importances.head(3).items():
            print(f"  - {feat}: {imp*100:.1f}%")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    run_honest_pipeline()