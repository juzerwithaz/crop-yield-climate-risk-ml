"""
12_stage3_yield_risk_classification.py

Stage 3: Yield-Risk Classification and Explainability
- Evaluates binary low-yield risk (High_Yield_Risk: Yield <= tau_crop) across 5 crops.
- Thresholds tau_15, tau_20, tau_25 calculated exclusively from training data (Year <= 2007).
- Excludes 35 suspected invalid zero-yield rows (from pandas sum of NaN Production).
- Compares Feature Group A (Climate & Soil) vs Feature Group B (Climate, Soil + Lag).
- Benchmarks against DummyClassifier baselines.
- Models: Logistic Regression, Random Forest, HistGradientBoosting (with class_weight='balanced').
- Handles Maize zero-validation-positive case cleanly (training chronological validation).
- Refits selected pipeline on combined development set (train + validation) before final test.
- Explains models via Permutation Importance and Logistic Regression coefficients.
- Post-hoc risk prevalence analysis across Stage 2 agro-environmental clusters.
- Sensitivity analyses: threshold variations (p15, p20, p25) and invalid zeros inclusion.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.inspection import permutation_importance

INPUT_STAGE1_CSV = "tn_master_dataset_stage1_ready.csv"
INPUT_STAGE2_CSV = os.path.join("stage2_results", "stage2_cluster_assignments.csv")
RESULTS_DIR = "stage3_results"

GROUP_A_FEATURES = ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K']
GROUP_B_FEATURES = ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K', 'Yield_Lag_Avg', 'Yield_Lag_Years_Available']

RANDOM_SEED = 42

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def calculate_training_thresholds(df_primary):
    print("Calculating training-only yield-risk thresholds (Year <= 2007)...")
    train_df = df_primary[df_primary['Split'] == 'train']
    
    thresholds = []
    for crop in sorted(df_primary['Crop'].unique()):
        y_tr = train_df[train_df['Crop'] == crop]['Yield']
        p15 = float(np.percentile(y_tr, 15))
        p20 = float(np.percentile(y_tr, 20))
        p25 = float(np.percentile(y_tr, 25))
        thresholds.append({
            'Crop': crop,
            'Train_N': len(y_tr),
            'Yield_Min': float(y_tr.min()),
            'Yield_Mean': float(y_tr.mean()),
            'Yield_Max': float(y_tr.max()),
            'Threshold_p15': p15,
            'Threshold_p20_Primary': p20,
            'Threshold_p25': p25
        })
    df_thresh = pd.DataFrame(thresholds)
    df_thresh.to_csv(os.path.join(RESULTS_DIR, "stage3_risk_thresholds.csv"), index=False)
    print("  Saved thresholds to: stage3_results/stage3_risk_thresholds.csv")
    return df_thresh

def compute_class_distributions(df_primary, df_thresh):
    print("Computing class distribution and lag availability...")
    records = []
    thresh_map = df_thresh.set_index('Crop')['Threshold_p20_Primary'].to_dict()
    
    for crop in sorted(df_primary['Crop'].unique()):
        t = thresh_map[crop]
        crop_df = df_primary[df_primary['Crop'] == crop].copy()
        crop_df['Risk'] = (crop_df['Yield'] <= t).astype(int)
        
        for split in ['train', 'validation', 'test']:
            sub = crop_df[crop_df['Split'] == split]
            tot = len(sub)
            n_pos = int(sub['Risk'].sum())
            n_neg = tot - n_pos
            prev = (n_pos / tot * 100) if tot > 0 else 0.0
            
            # Lag availability
            with_lag = int((sub['Yield_Lag_Years_Available'] > 0).sum())
            missing_lag = tot - with_lag
            
            records.append({
                'Crop': crop,
                'Split': split,
                'Total_Observations': tot,
                'High_Risk_Positives': n_pos,
                'Normal_Negatives': n_neg,
                'Prevalence_Pct': round(prev, 2),
                'Eligible_With_Lag': with_lag,
                'Missing_Lag': missing_lag,
                'Threshold_Used': t
            })
            
    df_dist = pd.DataFrame(records)
    df_dist.to_csv(os.path.join(RESULTS_DIR, "stage3_class_distribution.csv"), index=False)
    print("  Saved class distributions to: stage3_results/stage3_class_distribution.csv")
    return df_dist, thresh_map

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Computes comprehensive classification metrics.
    Gracefully handles edge cases where only one class is present.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    
    acc = float(accuracy_score(y_true, y_pred))
    
    # When both classes are present in true labels
    if n_pos > 0 and n_neg > 0:
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        
        if y_prob is not None:
            try:
                auc_roc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                auc_roc = np.nan
            try:
                auc_pr = float(average_precision_score(y_true, y_prob))
            except Exception:
                auc_pr = np.nan
        else:
            auc_roc = np.nan
            auc_pr = np.nan
            
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    else:
        # Edge case: only one class in y_true (e.g. Maize validation where n_pos == 0)
        bacc = np.nan
        prec = np.nan
        rec = np.nan
        f1 = np.nan
        auc_roc = np.nan
        auc_pr = np.nan
        tn = int((y_pred == 0).sum()) if n_neg > 0 else 0
        fp = int((y_pred == 1).sum()) if n_neg > 0 else 0
        fn = 0
        tp = 0
        
    return {
        'N': n, 'Positives': n_pos, 'Negatives': n_neg,
        'Accuracy': acc, 'Balanced_Accuracy': bacc,
        'Precision': prec, 'Recall': rec, 'F1': f1,
        'ROC_AUC': auc_roc, 'PR_AUC': auc_pr,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

def build_classifier(model_name, c_val=1.0):
    if model_name == 'Dummy_Prior':
        return DummyClassifier(strategy='prior')
    elif model_name == 'Dummy_Stratified':
        return DummyClassifier(strategy='stratified', random_state=RANDOM_SEED)
    elif model_name == 'LogisticRegression':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', C=c_val, random_state=RANDOM_SEED, max_iter=1000))
        ])
    elif model_name == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=3,
            class_weight='balanced', random_state=RANDOM_SEED
        )
    elif model_name == 'HistGradientBoosting':
        return HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_depth=3,
            class_weight='balanced', random_state=RANDOM_SEED
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_and_eval_classifier(model, X_tr, y_tr, X_ev, y_ev):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_ev)
    
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X_ev)[:, 1]
        except Exception:
            probs = None
    else:
        probs = None
        
    return compute_classification_metrics(y_ev, preds, probs), model, preds, probs

def run_stage3_pipeline():
    ensure_dir(RESULTS_DIR)
    print("=" * 80)
    print("STAGE 3: YIELD-RISK CLASSIFICATION AND EXPLAINABILITY")
    print("=" * 80)
    
    df_s1 = pd.read_csv(INPUT_STAGE1_CSV)
    zero_mask = (df_s1['Yield'] == 0.0)
    df_primary = df_s1[~zero_mask].copy()
    print(f"Primary analysis dataset: {len(df_primary)} observations (excluded 35 invalid zeros).")
    
    # 1. Thresholds & Distribution
    df_thresh = calculate_training_thresholds(df_primary)
    df_dist, thresh_map = compute_class_distributions(df_primary, df_thresh)
    
    # Add High_Yield_Risk target to df_primary
    df_primary['High_Yield_Risk'] = df_primary.apply(
        lambda r: int(r['Yield'] <= thresh_map[r['Crop']]), axis=1
    )
    
    crops = sorted(df_primary['Crop'].unique())
    validation_records = []
    selected_configurations = {}
    
    # Models to compare
    model_types = [
        ('Dummy_Prior', [None]),
        ('Dummy_Stratified', [None]),
        ('LogisticRegression', [0.1, 1.0, 10.0]),
        ('RandomForest', [None]),
        ('HistGradientBoosting', [None])
    ]
    
    # =========================================================================
    # PHASE 1: VALIDATION COMPARISON & MODEL SELECTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: VALIDATION EVALUATION (TRAIN <= 2007, VAL 2008-2009)")
    print("=" * 80)
    
    for crop in crops:
        crop_df = df_primary[df_primary['Crop'] == crop]
        train_df = crop_df[crop_df['Split'] == 'train']
        val_df = crop_df[crop_df['Split'] == 'validation']
        
        # Matched subsets with valid lag
        train_matched = train_df[train_df['Yield_Lag_Years_Available'] > 0]
        val_matched = val_df[val_df['Yield_Lag_Years_Available'] > 0]
        
        settings = [
            ('Group_A_Matched', GROUP_A_FEATURES, train_matched, val_matched),
            ('Group_B_Lag', GROUP_B_FEATURES, train_matched, val_matched),
            ('Group_A_All', GROUP_A_FEATURES, train_df, val_df)
        ]
        
        crop_candidates = []
        
        for setting_name, feat_cols, tr_sub, v_sub in settings:
            X_tr, y_tr = tr_sub[feat_cols], tr_sub['High_Yield_Risk']
            X_v, y_v = v_sub[feat_cols], v_sub['High_Yield_Risk']
            
            for m_name, c_vals in model_types:
                for c_val in c_vals:
                    clf = build_classifier(m_name, c_val=c_val if c_val is not None else 1.0)
                    metrics, fitted_model, _, _ = train_and_eval_classifier(clf, X_tr, y_tr, X_v, y_v)
                    
                    param_str = f"C={c_val}" if c_val is not None else "default"
                    rec = {
                        'Crop': crop, 'Model': m_name, 'Parameters': param_str,
                        'FeatureGroup': setting_name,
                        'Train_N': len(tr_sub), 'Val_N': len(v_sub),
                        'Val_Positives': metrics['Positives'],
                        'Accuracy': metrics['Accuracy'],
                        'Balanced_Accuracy': metrics['Balanced_Accuracy'],
                        'Precision': metrics['Precision'],
                        'Recall': metrics['Recall'],
                        'F1': metrics['F1'],
                        'ROC_AUC': metrics['ROC_AUC'],
                        'PR_AUC': metrics['PR_AUC'],
                        'TN': metrics['TN'], 'FP': metrics['FP'],
                        'FN': metrics['FN'], 'TP': metrics['TP']
                    }
                    validation_records.append(rec)
                    
                    # Store eligible models for selection (Group B and Group A Matched)
                    if setting_name in ['Group_A_Matched', 'Group_B_Lag'] and not m_name.startswith('Dummy'):
                        crop_candidates.append({
                            'Model': m_name, 'Parameters': param_str, 'C': c_val,
                            'FeatureGroup': setting_name, 'Features': feat_cols,
                            'Metrics': metrics
                        })
                        
        # -------------------------------------------------------------
        # Model Selection per Crop
        # -------------------------------------------------------------
        if crop == 'Maize':
            # Special documented case: 0 positives in validation split.
            # Perform training chronological validation (Train-early <= 2003 vs Train-late == 2004)
            print("  [Maize Special Selection]: 0 validation positives. Using training chronological validation (<=2003 vs 2004).")
            m_tr_early = train_matched[train_matched['Year'] <= 2003]
            m_tr_late = train_matched[train_matched['Year'] == 2004]
            
            maize_evals = []
            for cand in crop_candidates:
                m_clf = build_classifier(cand['Model'], c_val=cand['C'] if cand['C'] is not None else 1.0)
                m_metrics, _, _, _ = train_and_eval_classifier(
                    m_clf, m_tr_early[cand['Features']], m_tr_early['High_Yield_Risk'],
                    m_tr_late[cand['Features']], m_tr_late['High_Yield_Risk']
                )
                maize_evals.append({
                    'Config': cand,
                    'Chron_F1': m_metrics['F1'],
                    'Chron_BAcc': m_metrics['Balanced_Accuracy'],
                    'Chron_Rec': m_metrics['Recall']
                })
            df_m_eval = pd.DataFrame(maize_evals).sort_values(by=['Chron_F1', 'Chron_BAcc', 'Chron_Rec'], ascending=False)
            best_config = df_m_eval.iloc[0]['Config']
            best_config['Selection_Rationale'] = f"Chronological train split (<=2003 vs 2004) F1: {df_m_eval.iloc[0]['Chron_F1']:.4f}"
            selected_configurations[crop] = best_config
        else:
            # Standard crops: select on Validation F1, then Balanced Accuracy, then Recall
            df_cands = pd.DataFrame([{
                'Config': c,
                'F1': c['Metrics']['F1'],
                'BAcc': c['Metrics']['Balanced_Accuracy'],
                'Recall': c['Metrics']['Recall']
            } for c in crop_candidates])
            df_cands = df_cands.sort_values(by=['F1', 'BAcc', 'Recall'], ascending=False)
            best_config = df_cands.iloc[0]['Config']
            best_config['Selection_Rationale'] = f"Validation F1: {best_config['Metrics']['F1']:.4f}, BAcc: {best_config['Metrics']['Balanced_Accuracy']:.4f}"
            selected_configurations[crop] = best_config
            
        print(f"Selected Model for {crop:15s}: {best_config['Model']} ({best_config['FeatureGroup']}) | {best_config['Selection_Rationale']}")

    df_val_results = pd.DataFrame(validation_records)
    val_csv_path = os.path.join(RESULTS_DIR, "stage3_validation_comparison.csv")
    df_val_results.to_csv(val_csv_path, index=False)
    print(f"\nSaved validation comparison to: {val_csv_path}")

    # =========================================================================
    # PHASE 2: REFIT ON TRAIN + VALIDATION & FINAL TEST EVALUATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: REFIT ON TRAIN + VALIDATION & FINAL TEST EVALUATION (2010-2013)")
    print("=" * 80)
    
    test_records = []
    prediction_records = []
    final_fitted_pipelines = {}
    
    for crop in crops:
        crop_df = df_primary[df_primary['Crop'] == crop]
        best = selected_configurations[crop]
        feat_cols = best['Features']
        
        train_val_df = crop_df[crop_df['Split'].isin(['train', 'validation'])]
        test_df = crop_df[crop_df['Split'] == 'test']
        
        # Filter for matched lag
        if best['FeatureGroup'] in ['Group_B_Lag', 'Group_A_Matched']:
            dev_data = train_val_df[train_val_df['Yield_Lag_Years_Available'] > 0]
            eval_test = test_df[test_df['Yield_Lag_Years_Available'] > 0]
        else:
            dev_data = train_val_df
            eval_test = test_df
            
        X_dev, y_dev = dev_data[feat_cols], dev_data['High_Yield_Risk']
        X_test, y_test = eval_test[feat_cols], eval_test['High_Yield_Risk']
        
        # 1. Baseline: Dummy Prior
        dummy_prior = DummyClassifier(strategy='prior')
        m_dummy, _, _, _ = train_and_eval_classifier(dummy_prior, X_dev, y_dev, X_test, y_test)
        test_records.append({
            'Crop': crop, 'Model_Type': 'Baseline_DummyPrior',
            'Model': 'DummyPrior', 'FeatureGroup': best['FeatureGroup'],
            'Dev_N': len(dev_data), 'Test_N': len(eval_test),
            'Test_Positives': m_dummy['Positives'],
            'Accuracy': m_dummy['Accuracy'], 'Balanced_Accuracy': m_dummy['Balanced_Accuracy'],
            'Precision': m_dummy['Precision'], 'Recall': m_dummy['Recall'],
            'F1': m_dummy['F1'], 'ROC_AUC': m_dummy['ROC_AUC'], 'PR_AUC': m_dummy['PR_AUC'],
            'TN': m_dummy['TN'], 'FP': m_dummy['FP'], 'FN': m_dummy['FN'], 'TP': m_dummy['TP']
        })
        
        # 2. Selected Pipeline refitted on Dev
        final_model = build_classifier(best['Model'], c_val=best['C'] if best['C'] is not None else 1.0)
        m_selected, fitted_final, test_preds, test_probs = train_and_eval_classifier(
            final_model, X_dev, y_dev, X_test, y_test
        )
        final_fitted_pipelines[crop] = fitted_final
        
        test_records.append({
            'Crop': crop, 'Model_Type': 'Selected_Model',
            'Model': best['Model'], 'FeatureGroup': best['FeatureGroup'],
            'Dev_N': len(dev_data), 'Test_N': len(eval_test),
            'Test_Positives': m_selected['Positives'],
            'Accuracy': m_selected['Accuracy'], 'Balanced_Accuracy': m_selected['Balanced_Accuracy'],
            'Precision': m_selected['Precision'], 'Recall': m_selected['Recall'],
            'F1': m_selected['F1'], 'ROC_AUC': m_selected['ROC_AUC'], 'PR_AUC': m_selected['PR_AUC'],
            'TN': m_selected['TN'], 'FP': m_selected['FP'], 'FN': m_selected['FN'], 'TP': m_selected['TP']
        })
        
        print(f"\nFinal Test Results for {crop} ({best['Model']}):")
        print(f"  Test N: {len(eval_test)} | Positives: {m_selected['Positives']} ({m_selected['Positives']/len(eval_test)*100:.1f}%)")
        print(f"  Accuracy: {m_selected['Accuracy']:.4f} | Balanced Acc: {m_selected['Balanced_Accuracy']:.4f}")
        print(f"  Precision: {m_selected['Precision']:.4f} | Recall: {m_selected['Recall']:.4f} | F1: {m_selected['F1']:.4f}")
        print(f"  ROC-AUC: {m_selected['ROC_AUC']:.4f} | PR-AUC: {m_selected['PR_AUC']:.4f}")
        print(f"  Confusion Matrix: TN={m_selected['TN']}, FP={m_selected['FP']}, FN={m_selected['FN']}, TP={m_selected['TP']}")
        
        # Log test predictions
        for idx, row in eval_test.iterrows():
            pos_in_test = eval_test.index.get_loc(idx)
            prediction_records.append({
                'District': row['District'], 'Crop': crop, 'Year': row['Year'],
                'Split': 'test', 'Actual_Yield': row['Yield'],
                'True_Risk': row['High_Yield_Risk'],
                'Predicted_Risk': test_preds[pos_in_test],
                'Predicted_Risk_Prob': test_probs[pos_in_test] if test_probs is not None else np.nan,
                'Model': best['Model'], 'FeatureGroup': best['FeatureGroup']
            })
            
    df_test_metrics = pd.DataFrame(test_records)
    test_csv_path = os.path.join(RESULTS_DIR, "stage3_test_metrics.csv")
    df_test_metrics.to_csv(test_csv_path, index=False)
    print(f"\nSaved test metrics to: {test_csv_path}")
    
    df_test_preds = pd.DataFrame(prediction_records)
    preds_csv_path = os.path.join(RESULTS_DIR, "stage3_test_predictions.csv")
    df_test_preds.to_csv(preds_csv_path, index=False)
    print(f"Saved test predictions log to: {preds_csv_path}")

    # =========================================================================
    # PHASE 3: EXPLAINABILITY & FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: EXPLAINABILITY & FEATURE IMPORTANCE")
    print("=" * 80)
    
    importance_records = []
    
    for crop in crops:
        crop_df = df_primary[df_primary['Crop'] == crop]
        best = selected_configurations[crop]
        feat_cols = best['Features']
        
        train_matched = crop_df[(crop_df['Split'] == 'train') & (crop_df['Yield_Lag_Years_Available'] > 0)]
        val_matched = crop_df[(crop_df['Split'] == 'validation') & (crop_df['Yield_Lag_Years_Available'] > 0)]
        
        # Fit on train matched to compute permutation on validation
        clf_val = build_classifier(best['Model'], c_val=best['C'] if best['C'] is not None else 1.0)
        clf_val.fit(train_matched[feat_cols], train_matched['High_Yield_Risk'])
        
        # Check if validation has both classes
        val_pos = val_matched['High_Yield_Risk'].sum()
        if val_pos > 0:
            perm = permutation_importance(
                clf_val, val_matched[feat_cols], val_matched['High_Yield_Risk'],
                scoring='balanced_accuracy', n_repeats=10, random_state=RANDOM_SEED
            )
            for i, f in enumerate(feat_cols):
                importance_records.append({
                    'Crop': crop, 'Model': best['Model'], 'Feature': f,
                    'Permutation_Mean_Impact': float(perm.importances_mean[i]),
                    'Permutation_Std_Impact': float(perm.importances_std[i]),
                    'Metric_Used': 'Balanced_Accuracy',
                    'Validation_Status': 'Valid (Both classes present)'
                })
        else:
            # Maize case: validation has 0 positives
            print(f"  [Permutation Importance Note]: {crop} validation has 0 positive targets. Permutation computed on chronological train-late set (2004).")
            m_tr_late = train_matched[train_matched['Year'] == 2004]
            perm = permutation_importance(
                clf_val, m_tr_late[feat_cols], m_tr_late['High_Yield_Risk'],
                scoring='balanced_accuracy', n_repeats=10, random_state=RANDOM_SEED
            )
            for i, f in enumerate(feat_cols):
                importance_records.append({
                    'Crop': crop, 'Model': best['Model'], 'Feature': f,
                    'Permutation_Mean_Impact': float(perm.importances_mean[i]),
                    'Permutation_Std_Impact': float(perm.importances_std[i]),
                    'Metric_Used': 'Balanced_Accuracy',
                    'Validation_Status': 'Evaluated on Chronological Train-Late (2004)'
                })
                
        # If Logistic Regression was used, also record standardized coefficients / odds ratios
        if best['Model'] == 'LogisticRegression':
            pipe = final_fitted_pipelines[crop]
            coefs = pipe.named_steps['clf'].coef_[0]
            for i, f in enumerate(feat_cols):
                importance_records.append({
                    'Crop': crop, 'Model': 'LogisticRegression', 'Feature': f,
                    'Permutation_Mean_Impact': float(coefs[i]),
                    'Permutation_Std_Impact': float(np.exp(coefs[i])), # Odds Ratio
                    'Metric_Used': 'Standardized_Logit_Coeff_and_Odds_Ratio',
                    'Validation_Status': 'Model Parameter'
                })
                
    df_imp = pd.DataFrame(importance_records)
    imp_csv_path = os.path.join(RESULTS_DIR, "stage3_feature_importance.csv")
    df_imp.to_csv(imp_csv_path, index=False)
    print(f"Saved feature importance table to: {imp_csv_path}")

    # =========================================================================
    # PHASE 4: POST-HOC STAGE 2 CLUSTER RISK ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: POST-HOC CLUSTER RISK ANALYSIS")
    print("=" * 80)
    
    df_s2 = pd.read_csv(INPUT_STAGE2_CSV)
    df_merged_risk = pd.merge(df_primary, df_s2[['District', 'Cluster_ID']], on='District', how='left')
    
    cluster_risk_records = []
    for crop in crops:
        c_sub = df_merged_risk[df_merged_risk['Crop'] == crop]
        for cid in sorted(df_s2['Cluster_ID'].unique()):
            cl_sub = c_sub[c_sub['Cluster_ID'] == cid]
            n_obs = len(cl_sub)
            n_dist = cl_sub['District'].nunique()
            n_risk = int(cl_sub['High_Yield_Risk'].sum())
            rate = (n_risk / n_obs * 100) if n_obs > 0 else 0.0
            
            cluster_risk_records.append({
                'Crop': crop,
                'Cluster_ID': cid,
                'Cluster_Name': get_cluster_name(cid),
                'Total_Observations': n_obs,
                'Distinct_Districts': n_dist,
                'High_Risk_Observations': n_risk,
                'Empirical_Risk_Rate_Pct': round(rate, 2),
                'Notes': 'Descriptive empirical association (small sample per cluster)'
            })
            
    df_cl_risk = pd.DataFrame(cluster_risk_records)
    cl_risk_csv_path = os.path.join(RESULTS_DIR, "stage3_cluster_risk_summary.csv")
    df_cl_risk.to_csv(cl_risk_csv_path, index=False)
    print(f"Saved cluster risk summary to: {cl_risk_csv_path}")

    # =========================================================================
    # PHASE 5: SENSITIVITY ANALYSES (THRESHOLDS & INVALID ZEROS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: SENSITIVITY ANALYSES")
    print("=" * 80)
    
    sens_records = []
    
    # 1. Threshold Sensitivity: p15 and p25
    for p_name, p_val in [('Threshold_p15', 15), ('Threshold_p25', 25)]:
        for crop in crops:
            crop_df = df_primary[df_primary['Crop'] == crop].copy()
            tr_y = crop_df[crop_df['Split'] == 'train']['Yield']
            t_sens = float(np.percentile(tr_y, p_val))
            
            crop_df['Risk_Sens'] = (crop_df['Yield'] <= t_sens).astype(int)
            best = selected_configurations[crop]
            feat_cols = best['Features']
            
            dev_df = crop_df[crop_df['Split'].isin(['train', 'validation']) & (crop_df['Yield_Lag_Years_Available'] > 0)]
            test_df = crop_df[(crop_df['Split'] == 'test') & (crop_df['Yield_Lag_Years_Available'] > 0)]
            
            clf = build_classifier(best['Model'], c_val=best['C'] if best['C'] is not None else 1.0)
            m_sens, _, _, _ = train_and_eval_classifier(
                clf, dev_df[feat_cols], dev_df['Risk_Sens'], test_df[feat_cols], test_df['Risk_Sens']
            )
            
            sens_records.append({
                'Sensitivity_Type': 'Threshold_Percentile',
                'Crop': crop, 'Scenario': f'Percentile_{p_val}',
                'Threshold_Value': round(t_sens, 4),
                'Test_N': len(test_df),
                'Test_Positives': m_sens['Positives'],
                'Accuracy': m_sens['Accuracy'],
                'Balanced_Accuracy': m_sens['Balanced_Accuracy'],
                'Precision': m_sens['Precision'],
                'Recall': m_sens['Recall'],
                'F1': m_sens['F1'],
                'ROC_AUC': m_sens['ROC_AUC']
            })
            
    # 2. Invalid Zeros Sensitivity: Including the 35 invalid zeros
    print("  Evaluating sensitivity of including the 35 suspected invalid zero targets...")
    for crop in crops:
        crop_df_all = df_s1[df_s1['Crop'] == crop].copy()
        t20 = thresh_map[crop]
        crop_df_all['Risk_All'] = (crop_df_all['Yield'] <= t20).astype(int)
        
        best = selected_configurations[crop]
        feat_cols = best['Features']
        
        dev_df = crop_df_all[crop_df_all['Split'].isin(['train', 'validation']) & (crop_df_all['Yield_Lag_Years_Available'] > 0)]
        test_df = crop_df_all[(crop_df_all['Split'] == 'test') & (crop_df_all['Yield_Lag_Years_Available'] > 0)]
        
        clf = build_classifier(best['Model'], c_val=best['C'] if best['C'] is not None else 1.0)
        m_zero, _, _, _ = train_and_eval_classifier(
            clf, dev_df[feat_cols], dev_df['Risk_All'], test_df[feat_cols], test_df['Risk_All']
        )
        
        sens_records.append({
            'Sensitivity_Type': 'Invalid_Zeros_Included',
            'Crop': crop, 'Scenario': 'With_35_Suspected_Invalid_Zeros',
            'Threshold_Value': round(t20, 4),
            'Test_N': len(test_df),
            'Test_Positives': m_zero['Positives'],
            'Accuracy': m_zero['Accuracy'],
            'Balanced_Accuracy': m_zero['Balanced_Accuracy'],
            'Precision': m_zero['Precision'],
            'Recall': m_zero['Recall'],
            'F1': m_zero['F1'],
            'ROC_AUC': m_zero['ROC_AUC']
        })
        
    df_sens = pd.DataFrame(sens_records)
    sens_csv_path = os.path.join(RESULTS_DIR, "stage3_sensitivity_analysis.csv")
    df_sens.to_csv(sens_csv_path, index=False)
    print(f"Saved sensitivity analysis to: {sens_csv_path}")

    # =========================================================================
    # PHASE 6: GENERATE VISUALIZATIONS
    # =========================================================================
    generate_stage3_plots(df_dist, df_val_results, df_test_metrics, df_imp, df_cl_risk, df_sens)

def get_cluster_name(cid):
    names = {
        0: 'Central & Delta Semi-Arid Plains',
        1: 'Western Ghats & Highland Agro-Zone',
        2: 'North-Western Plateau',
        3: 'Coastal Humid Plain'
    }
    return names.get(cid, f'Cluster {cid}')

def generate_stage3_plots(df_dist, df_val_results, df_test_metrics, df_imp, df_cl_risk, df_sens):
    print("\nGenerating Stage 3 diagnostic plots...")
    
    # 1. Risk-Class Distribution by Crop and Split
    fig, ax = plt.subplots(figsize=(10, 5))
    df_piv = df_dist.pivot(index='Crop', columns='Split', values='Prevalence_Pct')[['train', 'validation', 'test']]
    df_piv.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', alpha=0.85)
    ax.set_ylabel('High-Risk Prevalence (%)')
    ax.set_title('High-Yield-Risk Prevalence (p20 Threshold) Across Chronological Splits')
    ax.axhline(20, color='r', linestyle='--', label='Expected Training Base (20%)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(title='Split')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage3_risk_distribution.png"), dpi=200)
    plt.close()
    
    # 2. Validation Model Comparison (Balanced Accuracy & F1)
    fig, ax = plt.subplots(figsize=(12, 5))
    val_sub = df_val_results[df_val_results['FeatureGroup'] == 'Group_B_Lag']
    crops = sorted(val_sub['Crop'].unique())
    models = [m for m in val_sub['Model'].unique() if not m.startswith('Dummy')]
    
    x = np.arange(len(crops))
    width = 0.25
    for i, m in enumerate(models):
        sub_m = val_sub[val_sub['Model'] == m]
        # Map values
        baccs = [sub_m[sub_m['Crop'] == c]['Balanced_Accuracy'].values[0] for c in crops]
        # Replace nan with 0 for plotting
        baccs_plot = [b if pd.notna(b) else 0.0 for b in baccs]
        ax.bar(x + (i - 1) * width, baccs_plot, width, label=m, edgecolor='black', alpha=0.85)
        
    ax.set_xticks(x)
    ax.set_xticklabels(crops)
    ax.set_ylabel('Validation Balanced Accuracy')
    ax.set_title('Validation Performance Comparison by Model Family (Group B Features)')
    ax.axhline(0.5, color='gray', linestyle=':', label='Chance (0.50)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage3_validation_models.png"), dpi=200)
    plt.close()
    
    # 3. Test Confusion Matrices for Selected Models
    crops = sorted(df_test_metrics[df_test_metrics['Model_Type'] == 'Selected_Model']['Crop'].unique())
    fig, axes = plt.subplots(1, len(crops), figsize=(18, 3.8))
    
    for i, crop in enumerate(crops):
        row = df_test_metrics[(df_test_metrics['Model_Type'] == 'Selected_Model') & (df_test_metrics['Crop'] == crop)].iloc[0]
        cm = np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]])
        
        ax = axes[i]
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(f"{crop}\n({row['Model']})\nF1: {row['F1']:.2f} | BAcc: {row['Balanced_Accuracy']:.2f}", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal (0)', 'Risk (1)'], fontsize=8)
        ax.set_yticklabels(['Normal (0)', 'Risk (1)'], fontsize=8)
        ax.set_xlabel('Predicted')
        if i == 0:
            ax.set_ylabel('Actual')
            
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                ax.text(c, r, str(val), ha='center', va='center',
                        color='white' if val > cm.max()/2 else 'black', fontweight='bold')
                
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage3_test_confusion_matrices.png"), dpi=200)
    plt.close()
    
    # 4. Feature Importance for Selected Models
    imp_sub = df_imp[df_imp['Metric_Used'] == 'Balanced_Accuracy']
    crops = sorted(imp_sub['Crop'].unique())
    fig, axes = plt.subplots(len(crops), 1, figsize=(9, 13))
    
    for i, crop in enumerate(crops):
        c_imp = imp_sub[imp_sub['Crop'] == crop].sort_values('Permutation_Mean_Impact')
        ax = axes[i]
        ax.barh(c_imp['Feature'], c_imp['Permutation_Mean_Impact'], xerr=c_imp['Permutation_Std_Impact'],
                color='#1f77b4', edgecolor='black', alpha=0.85, capsize=3)
        ax.axvline(0, color='black', lw=0.8)
        model_name = c_imp['Model'].iloc[0]
        ax.set_title(f"{crop}: Permutation Importance ({model_name}) — Impact on Balanced Accuracy", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage3_feature_importance.png"), dpi=200)
    plt.close()
    
    # 5. Risk Prevalence by Stage 2 Cluster Heatmap
    piv_risk = df_cl_risk.pivot(index='Cluster_Name', columns='Crop', values='Empirical_Risk_Rate_Pct')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(piv_risk.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=35)
    plt.colorbar(im, label='Empirical High-Risk Rate (%)')
    ax.set_xticks(np.arange(len(piv_risk.columns)))
    ax.set_xticklabels(piv_risk.columns, rotation=20)
    ax.set_yticks(np.arange(len(piv_risk.index)))
    ax.set_yticklabels(piv_risk.index, fontsize=9)
    ax.set_title('Empirical Yield-Risk Prevalence (%) Across Stage 2 Agro-Climatic Clusters')
    
    for r in range(len(piv_risk.index)):
        for c in range(len(piv_risk.columns)):
            val = piv_risk.values[r, c]
            ax.text(c, r, f"{val:.1f}%", ha='center', va='center',
                    color='white' if val > 20 else 'black', fontweight='bold', fontsize=9)
            
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage3_cluster_risk_heatmap.png"), dpi=200)
    plt.close()
    
    # 6. Threshold Sensitivity Comparison (F1 vs Percentile)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    thresh_sub = df_sens[df_sens['Sensitivity_Type'] == 'Threshold_Percentile']
    for crop in sorted(thresh_sub['Crop'].unique()):
        sub_c = thresh_sub[thresh_sub['Crop'] == crop].sort_values('Scenario')
        # Map scenario to percentile
        ax.plot(['p15', 'p25'], sub_c['Balanced_Accuracy'], marker='o', label=crop)
        
    ax.set_title('Test Balanced Accuracy Sensitivity to Training Risk Threshold (p15 vs p25)')
    ax.set_ylabel('Test Balanced Accuracy')
    ax.set_xlabel('Risk Definition Threshold Scenario')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage3_threshold_sensitivity.png"), dpi=200)
    plt.close()
    
    print("  All 6 plots successfully saved to stage3_results/.")

if __name__ == '__main__':
    run_stage3_pipeline()

