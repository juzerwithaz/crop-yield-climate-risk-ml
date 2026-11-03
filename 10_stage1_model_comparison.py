"""
10_stage1_model_comparison.py

Stage 1 Model Comparison Pipeline:
- Predict crop yield separately for 5 crops in Tamil Nadu.
- Predictor policy: Group A (Climate & Soil), Group B (Climate, Soil + Historical Lag).
- Excludes 35 suspected invalid zero-yield rows (from pandas sum of NaN Production).
- Evaluates baselines (Training Mean, Historical Yield Persistence).
- Trains Ridge, Random Forest, Gradient Boosting, HistGradientBoosting.
- Tests raw vs log1p target transformations on validation split (original yield scale).
- Selects best model per crop using validation split only (2008-2009).
- Refits selected pipeline on combined train + validation observations.
- Evaluates once on final test split (2010-2013).
- Generates diagnostic plots, sensitivity analysis, and CSV outputs.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

INPUT_CSV = "tn_master_dataset_stage1_ready.csv"
RESULTS_DIR = "stage1_results"

GROUP_A_FEATURES = ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K']
GROUP_B_FEATURES = ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K', 'Yield_Lag_Avg']

RANDOM_SEED = 42

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_and_audit_data():
    print("Loading prepared dataset:", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    
    # Identify 35 suspected invalid zero-yield observations
    zero_mask = (df['Yield'] == 0.0)
    n_zeros = zero_mask.sum()
    print(f"Identified {n_zeros} suspected invalid zero-yield observations (Production was NaN).")
    
    zero_breakdown = df[zero_mask].groupby(['Crop', 'Split']).size().unstack(fill_value=0)
    print("\nZero-yield breakdown by Crop and Split:")
    print(zero_breakdown.to_string())
    
    return df, zero_mask

def evaluate_predictions(y_true, y_pred):
    """Computes R2, RMSE, MAE on original yield scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Clip negative predictions at 0
    y_pred = np.clip(y_pred, 0, None)
    
    n = len(y_true)
    if n == 0:
        return {'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'N': 0}
    
    # Handle single value / zero variance edge case
    if np.var(y_true) == 0:
        r2 = 0.0
    else:
        r2 = r2_score(y_true, y_pred)
        
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae), 'N': int(n)}

def build_model(model_name, alpha=1.0):
    if model_name == 'Ridge':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=alpha, random_state=RANDOM_SEED))
        ])
    elif model_name == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=RANDOM_SEED
        )
    elif model_name == 'GradientBoosting':
        return GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_SEED
        )
    elif model_name == 'HistGradientBoosting':
        return HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_SEED
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_and_predict(model, X_train, y_train, X_eval, transform='raw'):
    """
    Fits pipeline on X_train, y_train. Applies transform if specified.
    Inverts prediction back to original yield scale.
    """
    if transform == 'log1p':
        y_train_fit = np.log1p(y_train)
    else:
        y_train_fit = y_train.copy()
        
    model.fit(X_train, y_train_fit)
    pred_fit = model.predict(X_eval)
    
    if transform == 'log1p':
        pred_orig = np.expm1(pred_fit)
    else:
        pred_orig = pred_fit
        
    return np.clip(pred_orig, 0, None)

def run_stage1_pipeline():
    ensure_dir(RESULTS_DIR)
    df, zero_mask = load_and_audit_data()
    
    # Filter for primary analysis
    df_primary = df[~zero_mask].copy()
    print(f"\nPrimary modeling dataset: {len(df_primary)} rows (excluded {zero_mask.sum()} invalid zeros).")
    
    crops = sorted(df_primary['Crop'].unique())
    validation_records = []
    
    # Store candidates per crop for validation selection
    crop_candidates = {}
    
    # =========================================================================
    # PHASE 1: VALIDATION EXPLORATION & MODEL SELECTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: VALIDATION COMPARISON (TRAIN <= 2007, VAL 2008-2009)")
    print("=" * 80)
    
    for crop in crops:
        crop_df = df_primary[df_primary['Crop'] == crop]
        train_df = crop_df[crop_df['Split'] == 'train']
        val_df = crop_df[crop_df['Split'] == 'validation']
        
        crop_candidates[crop] = []
        
        # 1. Eligible subsets:
        # A) All rows in crop split
        # B) Matched rows with Yield_Lag_Years_Available > 0
        train_matched = train_df[train_df['Yield_Lag_Years_Available'] > 0]
        val_matched = val_df[val_df['Yield_Lag_Years_Available'] > 0]
        
        # --- BASELINE 1: Training Mean ---
        # Evaluate on matched validation rows
        train_mean_val = train_matched['Yield'].mean()
        pred_b1_matched = np.full(len(val_matched), train_mean_val)
        m_b1_matched = evaluate_predictions(val_matched['Yield'], pred_b1_matched)
        rec_b1_matched = {
            'Crop': crop, 'Model': 'Baseline_TrainMean', 'FeatureGroup': 'Matched_Lag_Subset',
            'Transform': 'none', 'Alpha': np.nan,
            'Val_R2': m_b1_matched['R2'], 'Val_RMSE': m_b1_matched['RMSE'], 'Val_MAE': m_b1_matched['MAE'],
            'Train_N': len(train_matched), 'Val_N': len(val_matched)
        }
        validation_records.append(rec_b1_matched)
        
        # Also on all validation rows
        train_mean_all = train_df['Yield'].mean()
        pred_b1_all = np.full(len(val_df), train_mean_all)
        m_b1_all = evaluate_predictions(val_df['Yield'], pred_b1_all)
        rec_b1_all = {
            'Crop': crop, 'Model': 'Baseline_TrainMean', 'FeatureGroup': 'All_Rows',
            'Transform': 'none', 'Alpha': np.nan,
            'Val_R2': m_b1_all['R2'], 'Val_RMSE': m_b1_all['RMSE'], 'Val_MAE': m_b1_all['MAE'],
            'Train_N': len(train_df), 'Val_N': len(val_df)
        }
        validation_records.append(rec_b1_all)
        
        # --- BASELINE 2: Historical Yield Persistence (Yield_Lag_Avg) ---
        pred_b2 = val_matched['Yield_Lag_Avg'].values
        m_b2 = evaluate_predictions(val_matched['Yield'], pred_b2)
        rec_b2 = {
            'Crop': crop, 'Model': 'Baseline_HistoricalYield', 'FeatureGroup': 'Matched_Lag_Subset',
            'Transform': 'none', 'Alpha': np.nan,
            'Val_R2': m_b2['R2'], 'Val_RMSE': m_b2['RMSE'], 'Val_MAE': m_b2['MAE'],
            'Train_N': len(train_matched), 'Val_N': len(val_matched)
        }
        validation_records.append(rec_b2)
        
        # --- REGRESSION MODELS ---
        # Configurations to evaluate
        models_to_test = [
            ('Ridge', [0.1, 1.0, 10.0, 100.0]),
            ('RandomForest', [None]),
            ('GradientBoosting', [None]),
            ('HistGradientBoosting', [None])
        ]
        
        # Group comparisons:
        # Setting 1: Group A on Matched Subset (apples-to-apples with Group B)
        # Setting 2: Group B on Matched Subset (Climate + Soil + Lag)
        # Setting 3: Group A on All Eligible rows
        settings = [
            ('Group_A_Matched', GROUP_A_FEATURES, train_matched, val_matched),
            ('Group_B_Lag', GROUP_B_FEATURES, train_matched, val_matched),
            ('Group_A_All', GROUP_A_FEATURES, train_df, val_df)
        ]
        
        for setting_name, feature_list, tr_data, v_data in settings:
            X_tr = tr_data[feature_list]
            y_tr = tr_data['Yield']
            X_v = v_data[feature_list]
            y_v = v_data['Yield']
            
            for model_name, alphas in models_to_test:
                for alpha in alphas:
                    for transform in ['raw', 'log1p']:
                        mdl = build_model(model_name, alpha=alpha if alpha is not None else 1.0)
                        preds = train_and_predict(mdl, X_tr, y_tr, X_v, transform=transform)
                        metrics = evaluate_predictions(y_v, preds)
                        
                        rec = {
                            'Crop': crop,
                            'Model': model_name,
                            'FeatureGroup': setting_name,
                            'Transform': transform,
                            'Alpha': alpha if alpha is not None else np.nan,
                            'Val_R2': metrics['R2'],
                            'Val_RMSE': metrics['RMSE'],
                            'Val_MAE': metrics['MAE'],
                            'Train_N': len(tr_data),
                            'Val_N': len(v_data)
                        }
                        validation_records.append(rec)
                        
                        # Only consider models trained on eligible primary settings for selection
                        # (Group B or Group A Matched)
                        if setting_name in ['Group_A_Matched', 'Group_B_Lag']:
                            crop_candidates[crop].append({
                                'Model': model_name,
                                'FeatureGroup': setting_name,
                                'Features': feature_list,
                                'Transform': transform,
                                'Alpha': alpha,
                                'Val_R2': metrics['R2'],
                                'Val_RMSE': metrics['RMSE'],
                                'Val_MAE': metrics['MAE']
                            })
                            
    df_val_results = pd.DataFrame(validation_records)
    val_csv_path = os.path.join(RESULTS_DIR, "stage1_validation_comparison.csv")
    df_val_results.to_csv(val_csv_path, index=False)
    print(f"Saved validation comparison table ({len(df_val_results)} rows) to {val_csv_path}")

    # =========================================================================
    # PHASE 2: MODEL SELECTION PER CROP
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: MODEL SELECTION PER CROP (BASED ON VALIDATION R2)")
    print("=" * 80)
    
    selected_models = {}
    for crop in crops:
        cands = pd.DataFrame(crop_candidates[crop])
        # Sort by Val_R2 desc, then Val_RMSE asc
        cands = cands.sort_values(by=['Val_R2', 'Val_RMSE'], ascending=[False, True])
        best = cands.iloc[0].to_dict()
        selected_models[crop] = best
        
        print(f"\nSelected Model for {crop}:")
        print(f"  Model Family:    {best['Model']} (alpha={best['Alpha']})")
        print(f"  Feature Group:   {best['FeatureGroup']}")
        print(f"  Transformation:  {best['Transform']}")
        print(f"  Validation R2:   {best['Val_R2']:.4f} | RMSE: {best['Val_RMSE']:.4f} | MAE: {best['Val_MAE']:.4f}")

    # =========================================================================
    # PHASE 3: REFIT ON TRAIN + VALIDATION & FINAL TEST EVALUATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: REFIT ON TRAIN + VALIDATION & EVALUATE ON TEST (2010-2013)")
    print("=" * 80)
    
    test_records = []
    prediction_records = []
    
    for crop in crops:
        crop_df = df_primary[df_primary['Crop'] == crop]
        best = selected_models[crop]
        
        # Data splits
        train_val_df = crop_df[crop_df['Split'].isin(['train', 'validation'])]
        test_df = crop_df[crop_df['Split'] == 'test']
        
        # Filter for matched lag availability if chosen setting requires lag
        requires_lag = (best['FeatureGroup'] == 'Group_B_Lag')
        if requires_lag or best['FeatureGroup'] == 'Group_A_Matched':
            dev_data = train_val_df[train_val_df['Yield_Lag_Years_Available'] > 0]
            eval_test = test_df[test_df['Yield_Lag_Years_Available'] > 0]
        else:
            dev_data = train_val_df
            eval_test = test_df
            
        feat_cols = best['Features']
        X_dev = dev_data[feat_cols]
        y_dev = dev_data['Yield']
        X_test = eval_test[feat_cols]
        y_test = eval_test['Yield']
        
        # --- BASELINE 1 on Test (Mean of Dev Data) ---
        mean_dev_yield = y_dev.mean()
        pred_b1_test = np.full(len(eval_test), mean_dev_yield)
        m_b1_test = evaluate_predictions(y_test, pred_b1_test)
        test_records.append({
            'Crop': crop, 'Model_Type': 'Baseline_DevMean',
            'Model': 'DevMean', 'FeatureGroup': 'Matched_Subset', 'Transform': 'none',
            'Test_R2': m_b1_test['R2'], 'Test_RMSE': m_b1_test['RMSE'], 'Test_MAE': m_b1_test['MAE'],
            'Dev_N': len(dev_data), 'Test_N': len(eval_test)
        })
        
        # --- BASELINE 2 on Test (Historical Persistence) ---
        pred_b2_test = eval_test['Yield_Lag_Avg'].values
        m_b2_test = evaluate_predictions(y_test, pred_b2_test)
        test_records.append({
            'Crop': crop, 'Model_Type': 'Baseline_HistoricalYield',
            'Model': 'Yield_Lag_Avg', 'FeatureGroup': 'Matched_Subset', 'Transform': 'none',
            'Test_R2': m_b2_test['R2'], 'Test_RMSE': m_b2_test['RMSE'], 'Test_MAE': m_b2_test['MAE'],
            'Dev_N': len(dev_data), 'Test_N': len(eval_test)
        })
        
        # --- SELECTED MODEL ---
        final_pipeline = build_model(best['Model'], alpha=best['Alpha'] if pd.notna(best['Alpha']) else 1.0)
        
        # Refit on dev (train + validation)
        if best['Transform'] == 'log1p':
            y_dev_fit = np.log1p(y_dev)
        else:
            y_dev_fit = y_dev.copy()
            
        final_pipeline.fit(X_dev, y_dev_fit)
        
        # Predict on Test
        test_pred_fit = final_pipeline.predict(X_test)
        if best['Transform'] == 'log1p':
            test_preds = np.expm1(test_pred_fit)
        else:
            test_preds = test_pred_fit
        test_preds = np.clip(test_preds, 0, None)
        
        m_selected_test = evaluate_predictions(y_test, test_preds)
        test_records.append({
            'Crop': crop, 'Model_Type': 'Selected_Model',
            'Model': best['Model'], 'FeatureGroup': best['FeatureGroup'], 'Transform': best['Transform'],
            'Test_R2': m_selected_test['R2'], 'Test_RMSE': m_selected_test['RMSE'], 'Test_MAE': m_selected_test['MAE'],
            'Dev_N': len(dev_data), 'Test_N': len(eval_test)
        })
        
        print(f"\nTest Performance for {crop} ({best['Model']}):")
        print(f"  Test R2:   {m_selected_test['R2']:.4f} (Baseline Mean: {m_b1_test['R2']:.4f}, Baseline Persistence: {m_b2_test['R2']:.4f})")
        print(f"  Test RMSE: {m_selected_test['RMSE']:.4f} (Baseline Persistence: {m_b2_test['RMSE']:.4f})")
        print(f"  Test MAE:  {m_selected_test['MAE']:.4f} (Baseline Persistence: {m_b2_test['MAE']:.4f})")
        
        # Also predict on Validation for full predictions logging and plotting
        val_sub = crop_df[crop_df['Split'] == 'validation']
        if requires_lag or best['FeatureGroup'] == 'Group_A_Matched':
            val_sub = val_sub[val_sub['Yield_Lag_Years_Available'] > 0]
        val_pred_fit = final_pipeline.predict(val_sub[feat_cols])
        val_preds = np.expm1(val_pred_fit) if best['Transform'] == 'log1p' else val_pred_fit
        val_preds = np.clip(val_preds, 0, None)
        
        # Save predictions
        for idx, row in eval_test.iterrows():
            p_val = test_preds[eval_test.index.get_loc(idx)]
            prediction_records.append({
                'District': row['District'], 'Crop': crop, 'Year': row['Year'],
                'Split': 'test', 'Actual_Yield': row['Yield'], 'Predicted_Yield': p_val,
                'Residual': row['Yield'] - p_val,
                'Model': best['Model'], 'FeatureGroup': best['FeatureGroup']
            })
            
        for idx, row in val_sub.iterrows():
            p_val = val_preds[val_sub.index.get_loc(idx)]
            prediction_records.append({
                'District': row['District'], 'Crop': crop, 'Year': row['Year'],
                'Split': 'validation', 'Actual_Yield': row['Yield'], 'Predicted_Yield': p_val,
                'Residual': row['Yield'] - p_val,
                'Model': best['Model'], 'FeatureGroup': best['FeatureGroup']
            })
            
        # =========================================================================
        # PHASE 4: DIAGNOSTIC PLOTS & FEATURE IMPORTANCE
        # =========================================================================
        plot_crop_diagnostics(crop, y_test, test_preds, final_pipeline, feat_cols, best)
        
    df_test_results = pd.DataFrame(test_records)
    test_csv_path = os.path.join(RESULTS_DIR, "stage1_test_metrics.csv")
    df_test_results.to_csv(test_csv_path, index=False)
    print(f"\nSaved test metrics to {test_csv_path}")
    
    df_preds = pd.DataFrame(prediction_records)
    pred_csv_path = os.path.join(RESULTS_DIR, "stage1_predictions.csv")
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"Saved predictions log to {pred_csv_path}")

    # =========================================================================
    # PHASE 5: SENSITIVITY ANALYSIS (INCLUDING ZERO-YIELDS)
    # =========================================================================
    run_zero_yield_sensitivity(df, selected_models)

def plot_crop_diagnostics(crop, y_test, test_preds, model_pipeline, features, best_config):
    """Generates actual vs predicted, residuals, and feature importance plots."""
    crop_clean = crop.replace('(', '_').replace(')', '').replace(' ', '_').lower()
    
    # 1. Actual vs Predicted Scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, test_preds, alpha=0.7, edgecolors='k', color='#1f77b4')
    min_val = min(y_test.min(), test_preds.min())
    max_val = max(y_test.max(), test_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='1:1 Line')
    ax.set_xlabel('Actual Yield (t/ha)')
    ax.set_ylabel('Predicted Yield (t/ha)')
    ax.set_title(f'{crop}: Actual vs Predicted (Test 2010-2013)\nModel: {best_config["Model"]} ({best_config["FeatureGroup"]})')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"actual_vs_predicted_{crop_clean}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    # 2. Residual Distribution & Plot
    residuals = y_test - test_preds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(test_preds, residuals, alpha=0.7, edgecolors='k', color='#2ca02c')
    ax1.axhline(0, color='r', linestyle='--', lw=1.5)
    ax1.set_xlabel('Predicted Yield (t/ha)')
    ax1.set_ylabel('Residual (Actual - Predicted)')
    ax1.set_title(f'{crop}: Residuals vs Fitted')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.hist(residuals, bins=15, color='#2ca02c', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='r', linestyle='--', lw=1.5)
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{crop}: Residual Distribution')
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    res_path = os.path.join(RESULTS_DIR, f"residuals_{crop_clean}.png")
    plt.savefig(res_path, dpi=200)
    plt.close()
    
    # 3. Feature Importance
    fig, ax = plt.subplots(figsize=(7, 4.5))
    model_name = best_config['Model']
    
    if model_name == 'Ridge':
        reg = model_pipeline.named_steps['regressor']
        importances = reg.coef_
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, align='center', color='#ff7f0e')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Standardized Ridge Coefficient')
        ax.set_title(f'{crop}: Ridge Feature Coefficients')
    else:
        # Tree model
        if hasattr(model_pipeline, 'feature_importances_'):
            importances = model_pipeline.feature_importances_
        else:
            importances = np.zeros(len(features))
        idx = np.argsort(importances)
        ax.barh(np.arange(len(features)), importances[idx], align='center', color='#1f77b4')
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels([features[i] for i in idx])
        ax.set_xlabel('Gini / Tree Feature Importance')
        ax.set_title(f'{crop}: Feature Importance ({model_name})')
        
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    imp_path = os.path.join(RESULTS_DIR, f"feature_importance_{crop_clean}.png")
    plt.savefig(imp_path, dpi=200)
    plt.close()

def run_zero_yield_sensitivity(df, selected_models):
    """
    Sensitivity analysis: Evaluates how the selected models would perform
    if the 35 suspected invalid zero-yield rows were retained in validation and testing.
    """
    print("\n" + "=" * 80)
    print("PHASE 5: SENSITIVITY ANALYSIS (INCLUDING ZERO-YIELDS)")
    print("=" * 80)
    
    crops = sorted(df['Crop'].unique())
    sensitivity_records = []
    
    for crop in crops:
        crop_df = df[df['Crop'] == crop]
        best = selected_models[crop]
        feat_cols = best['Features']
        
        # Dev set including zeros
        dev_data = crop_df[crop_df['Split'].isin(['train', 'validation'])]
        test_data = crop_df[crop_df['Split'] == 'test']
        
        if best['FeatureGroup'] in ['Group_B_Lag', 'Group_A_Matched']:
            dev_data = dev_data[dev_data['Yield_Lag_Years_Available'] > 0]
            test_data = test_data[test_data['Yield_Lag_Years_Available'] > 0]
            
        final_pipeline = build_model(best['Model'], alpha=best['Alpha'] if pd.notna(best['Alpha']) else 1.0)
        
        y_dev = dev_data['Yield']
        y_dev_fit = np.log1p(y_dev) if best['Transform'] == 'log1p' else y_dev
        final_pipeline.fit(dev_data[feat_cols], y_dev_fit)
        
        # Test predictions
        y_test = test_data['Yield']
        test_pred_fit = final_pipeline.predict(test_data[feat_cols])
        test_preds = np.expm1(test_pred_fit) if best['Transform'] == 'log1p' else test_pred_fit
        test_preds = np.clip(test_preds, 0, None)
        
        m_sens = evaluate_predictions(y_test, test_preds)
        sensitivity_records.append({
            'Crop': crop,
            'Model': best['Model'],
            'Analysis': 'With_Invalid_Zeros_Included',
            'Test_R2': m_sens['R2'],
            'Test_RMSE': m_sens['RMSE'],
            'Test_MAE': m_sens['MAE'],
            'Dev_N': len(dev_data),
            'Test_N': len(test_data)
        })
        
    df_sens = pd.DataFrame(sensitivity_records)
    sens_path = os.path.join(RESULTS_DIR, "stage1_zero_yield_sensitivity.csv")
    df_sens.to_csv(sens_path, index=False)
    print(f"Saved zero-yield sensitivity metrics to {sens_path}")

if __name__ == '__main__':
    run_stage1_pipeline()

