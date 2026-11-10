import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def generate_final_report_plots():
    print("Loading data...")
    df = pd.read_csv("tn_master_dataset_clean.csv")

    # 1. Sort and Create Leak-Free Lag Feature
    # Sort carefully to ensure chronology
    df = df.sort_values(by=['District', 'Crop', 'Year']).reset_index(drop=True)
    
    # Create the 3-year rolling average of PREVIOUS years (shift 1)
    df['Yield_Lag_Avg'] = df.groupby(['District', 'Crop'])['Yield'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # Drop rows where lag is NaN (the very first recorded year for each district+crop combo)
    df = df.dropna(subset=['Yield_Lag_Avg']).copy()

    # 2. Setup Features and Target
    df['Log_Yield'] = np.log1p(df['Yield'])
    features = ['Area', 'Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K', 'Yield_Lag_Avg']
    crops = sorted(df['Crop'].unique())

    # Initialize plotting grid for Scatter Plots
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
    axes1 = axes1.flatten()
    importances_list = []

    print("Training models and building charts...")
    for i, crop in enumerate(crops):
        crop_df = df[df['Crop'] == crop]
        
        # 3. Train/Test Split (Year <= 2009 / Year >= 2010)
        train_df = crop_df[crop_df['Year'] <= 2009]
        test_df = crop_df[crop_df['Year'] >= 2010]
        
        X_train = train_df[features]
        y_train_log = train_df['Log_Yield']
        X_test = test_df[features]
        y_test_orig = test_df['Yield']
        
        # Train Random Forest
        rf = RandomForestRegressor(random_state=42, n_estimators=100, min_samples_leaf=3, max_depth=12)
        rf.fit(X_train, y_train_log)
        
        # Predict & Inverse Transform (expm1)
        y_pred = np.expm1(rf.predict(X_test))
        r2 = r2_score(y_test_orig, y_pred)
        
        # 4. Scatter Plot Construction
        ax = axes1[i]
        max_val = max(y_test_orig.max(), y_pred.max())
        
        # Red dashed perfect prediction line
        ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', alpha=0.6, label='Perfect Prediction')
        # Actual vs Predicted scatter
        ax.scatter(y_test_orig, y_pred, color='royalblue', edgecolor='black', alpha=0.7, s=40)
        
        ax.set_title(f"{crop}\nR² = {r2:.2f}")
        ax.set_xlabel("Actual Yield (tonnes/ha)")
        ax.set_ylabel("Predicted Yield (tonnes/ha)")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 5. Extract Feature Importances
        imp_df = pd.DataFrame({
            'Crop': crop,
            'Feature': features,
            'Importance': rf.feature_importances_ * 100
        }).sort_values(by='Importance', ascending=False).head(5)
        importances_list.append(imp_df)

    # Finalize and Save Scatter Plots
    axes1[5].axis('off') # Hide the empty 6th subplot
    fig1.suptitle("Stage 1 Regression: Actual vs Predicted Yield (Climate + Historical Trend Model, Test: 2010-2013)", 
                 fontsize=16, fontweight='bold', y=1.02)
    fig1.tight_layout()
    fig1.savefig("presentation_graphs_final.png", dpi=300, bbox_inches='tight')
    print("Saved -> presentation_graphs_final.png")

    # 6. Setup plotting grid for Feature Importances
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    axes2 = axes2.flatten()
    all_importances = pd.concat(importances_list)

    for i, crop in enumerate(crops):
        ax = axes2[i]
        crop_imp = all_importances[all_importances['Crop'] == crop]
        
        # Horizontal bar chart for readability
        sns.barplot(x='Importance', y='Feature', data=crop_imp, ax=ax, palette='viridis')
        
        ax.set_title(f"{crop}: Top 5 Features", fontweight='bold')
        ax.set_xlabel("Importance (%)")
        ax.set_ylabel("")
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)

    # Finalize and Save Feature Importance Plots
    axes2[5].axis('off')
    fig2.suptitle("Top 5 Feature Importances by Crop (Climate + Historical Trend Model)", 
                 fontsize=16, fontweight='bold', y=1.02)
    fig2.tight_layout()
    fig2.savefig("feature_importance_final.png", dpi=300, bbox_inches='tight')
    print("Saved -> feature_importance_final.png")

if __name__ == "__main__":
    generate_final_report_plots()