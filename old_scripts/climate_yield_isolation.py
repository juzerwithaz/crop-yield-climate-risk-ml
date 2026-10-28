import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_climate_isolated_pipeline():
    print("Loading tn_master_dataset_clean.csv...")
    df = pd.read_csv("tn_master_dataset_clean.csv")

    # The features we want the model to actually learn from
    features = ['Area', 'Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K']
    target = 'Yield'

    unique_crops = ['Rice', 'Maize', 'Sugarcane', 'Groundnut', 'Cotton(lint)']
    
    # Dictionary to store R² scores for the final summary table
    r2_summary = {}

    print("\n" + "="*60)
    print(" CLIMATE-ISOLATED MODELS (Year Excluded from Features)")
    print("="*60)

    for crop in unique_crops:
        crop_df = df[df['Crop'] == crop].copy()
        
        # 1. Use Year ONLY for splitting
        train_df = crop_df[crop_df['Year'] <= 2009]
        test_df = crop_df[crop_df['Year'] >= 2010]
        
        # 2. Extract features (excluding Year) and apply log1p to Target
        X_train = train_df[features]
        y_train_log = np.log1p(train_df[target])
        
        X_test = test_df[features]
        y_test_orig = test_df[target]  # Keep original for accurate metric calculation
        
        # 3. Train the model (using some regularization to prevent overfitting)
        rf = RandomForestRegressor(
            random_state=42, 
            n_estimators=100, 
            min_samples_leaf=3, 
            max_depth=12
        )
        rf.fit(X_train, y_train_log)
        
        # 4. Predict and inverse-transform (expm1)
        y_pred_log = rf.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        
        # 5. Compute Metrics
        rmse = mean_squared_error(y_test_orig, y_pred) ** 0.5
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        
        r2_summary[crop] = r2
        
        # 6. Extract Top 5 Feature Importances
        importances = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_ * 100
        }).sort_values(by='Importance', ascending=False).head(5)
        
        # Print Crop Results
        print(f"\n--- {crop.upper()} ---")
        print(f"Samples  : Train = {len(X_train)} | Test = {len(X_test)}")
        print(f"Metrics  : RMSE = {rmse:.2f} | MAE = {mae:.2f} | R² = {r2:.4f}")
        print("Top 5 Features:")
        for _, row in importances.iterrows():
            print(f"  - {row['Feature']:<10}: {row['Importance']:.1f}%")

    # 7. Print Final Summary Table
    print("\n" + "="*60)
    print(" R² SUMMARY TABLE (Test Set: 2010+)")
    print("="*60)
    print(f"{'Crop':<15} | {'R² Score'}")
    print("-" * 30)
    for crop, r2 in sorted(r2_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"{crop:<15} | {r2:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_climate_isolated_pipeline()