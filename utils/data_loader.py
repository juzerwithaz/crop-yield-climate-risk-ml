"""
data_loader.py
Cached data access layer for datasets, audit reports, and model metadata.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_DATASET_PATH = os.path.join(BASE_DIR, "tn_master_dataset_stage1_ready_v2.csv")
AUDIT_RESULTS_V2_DIR = os.path.join(BASE_DIR, "ml_audit_results_v2")
ARTIFACTS_V2_DIR = os.path.join(BASE_DIR, "model_artifacts_v2")
STAGE3_RESULTS_DIR = os.path.join(BASE_DIR, "stage3_results")

@st.cache_data
def load_v2_dataset() -> pd.DataFrame:
    """Loads the authoritative cleansed V2 master dataset."""
    if not os.path.exists(V2_DATASET_PATH):
        raise FileNotFoundError(f"V2 Dataset file not found: {V2_DATASET_PATH}")
    return pd.read_csv(V2_DATASET_PATH)

@st.cache_data
def load_regression_comparison() -> pd.DataFrame:
    """Loads the V1 vs V2 regression comparison metrics."""
    fpath = os.path.join(AUDIT_RESULTS_V2_DIR, "stage1_regression_comparison_v1_vs_v2.csv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Regression comparison CSV missing: {fpath}")
    return pd.read_csv(fpath)

@st.cache_data
def load_classification_comparison() -> pd.DataFrame:
    """Loads the V1 vs V2 risk-classification comparison metrics."""
    fpath = os.path.join(AUDIT_RESULTS_V2_DIR, "stage3_classification_comparison_v1_vs_v2.csv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Classification comparison CSV missing: {fpath}")
    return pd.read_csv(fpath)

@st.cache_data
def load_risk_thresholds_comparison() -> pd.DataFrame:
    """Loads the V1 vs V2 risk threshold comparison."""
    fpath = os.path.join(AUDIT_RESULTS_V2_DIR, "risk_thresholds_comparison_v1_vs_v2.csv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Risk thresholds comparison CSV missing: {fpath}")
    return pd.read_csv(fpath)

@st.cache_data
def load_v1_feature_importance_benchmark() -> pd.DataFrame:
    """
    Loads historical V1 permutation feature importances from Stage 3.
    Provenance note: Generated in Stage 3 on pre-cleansed data.
    """
    fpath = os.path.join(STAGE3_RESULTS_DIR, "stage3_feature_importance.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df['Provenance'] = 'V1_Stage3_Historical_Benchmark'
        return df
    return pd.DataFrame()

@st.cache_data
def extract_v2_model_weights_summary() -> dict:
    """
    Directly extracts feature weights / Gini importances from genuine serialized V2 models.
    Returns a dictionary of DataFrames keyed by crop and stage.
    """
    import joblib
    meta_path = os.path.join(ARTIFACTS_V2_DIR, "model_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    v2_weights = {}
    crops = ['Cotton(lint)', 'Groundnut', 'Maize', 'Rice', 'Sugarcane']
    features_group_b = ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K', 'Yield_Lag_Avg']

    for crop in crops:
        crop_id = meta['crops'][crop]['crop_identifier']

        # Regression Primary
        reg_file = os.path.join(ARTIFACTS_V2_DIR, f"yield_model_{crop_id}.joblib")
        if os.path.exists(reg_file):
            model = joblib.load(reg_file)
            weights = None
            method_desc = ""

            if hasattr(model, 'feature_importances_'):
                weights = model.feature_importances_
                method_desc = "Gini / Tree Feature Importance"
            elif hasattr(model, 'named_steps'):
                last_step = model.steps[-1][1]
                if hasattr(last_step, 'coef_'):
                    weights = np.abs(last_step.coef_)
                    method_desc = "Absolute Standardized Regression Coefficient"
                elif hasattr(last_step, 'feature_importances_'):
                    weights = last_step.feature_importances_
                    method_desc = "Gini / Tree Feature Importance"

            if weights is not None and len(weights) == len(features_group_b):
                # Normalize to percentage
                total = np.sum(weights)
                norm_pct = (weights / total * 100) if total > 0 else weights
                v2_weights[f"{crop}_regression"] = pd.DataFrame({
                    'Feature': features_group_b,
                    'Weight_Pct': norm_pct,
                    'Raw_Weight': weights,
                    'Method': method_desc,
                    'Model_Family': meta['crops'][crop]['regression']['model_family'],
                    'Artifact_Version': 'V2 (Genuine Retrained Model)'
                }).sort_values(by='Weight_Pct', ascending=False).reset_index(drop=True)

        # Classification Primary
        cls_file = os.path.join(ARTIFACTS_V2_DIR, f"risk_model_{crop_id}.joblib")
        if os.path.exists(cls_file):
            model = joblib.load(cls_file)
            weights = None
            method_desc = ""

            if hasattr(model, 'feature_importances_'):
                weights = model.feature_importances_
                method_desc = "Gini / Tree Feature Importance"
            elif hasattr(model, 'named_steps'):
                last_step = model.steps[-1][1]
                if hasattr(last_step, 'coef_'):
                    weights = np.abs(last_step.coef_[0])
                    method_desc = "Absolute Standardized Logit Coefficient"
                elif hasattr(last_step, 'feature_importances_'):
                    weights = last_step.feature_importances_
                    method_desc = "Gini / Tree Feature Importance"

            if weights is not None and len(weights) == len(features_group_b):
                total = np.sum(weights)
                norm_pct = (weights / total * 100) if total > 0 else weights
                v2_weights[f"{crop}_classification"] = pd.DataFrame({
                    'Feature': features_group_b,
                    'Weight_Pct': norm_pct,
                    'Raw_Weight': weights,
                    'Method': method_desc,
                    'Model_Family': meta['crops'][crop]['classification']['model_family'],
                    'Artifact_Version': 'V2 (Genuine Retrained Model)'
                }).sort_values(by='Weight_Pct', ascending=False).reset_index(drop=True)

    return v2_weights

@st.cache_data
def get_dataset_summary_stats() -> dict:
    """Computes exact dataset provenance and record counts from the V2 dataset."""
    df = load_v2_dataset()
    total_records = len(df)
    flag_counts = df['Data_Quality_Flag'].value_counts().to_dict()
    split_counts = df['Split'].value_counts().to_dict()

    valid_count = flag_counts.get('VALID', 0)
    scaled_count = flag_counts.get('CLEANSED_UNIT_SCALED', 0)
    excluded_missing = flag_counts.get('EXCLUDED_SOURCE_MISSING_PRODUCTION', 0)
    excluded_bifurcation = flag_counts.get('EXCLUDED_ADMIN_BIFURCATION', 0)

    eligible_count = valid_count + scaled_count
    total_excluded = excluded_missing + excluded_bifurcation

    districts = sorted(df['District'].unique().tolist())
    crops = sorted(df['Crop'].unique().tolist())
    years = [int(df['Year'].min()), int(df['Year'].max())]

    return {
        'total_records': total_records,
        'eligible_records': eligible_count,
        'valid_records': valid_count,
        'scaled_records': scaled_count,
        'excluded_missing_prod': excluded_missing,
        'excluded_admin_bifurcation': excluded_bifurcation,
        'total_excluded': total_excluded,
        'train_records': split_counts.get('train', 0),
        'validation_records': split_counts.get('validation', 0),
        'test_records': split_counts.get('test', 0),
        'districts_count': len(districts),
        'districts': districts,
        'crops_count': len(crops),
        'crops': crops,
        'years_range': years
    }

