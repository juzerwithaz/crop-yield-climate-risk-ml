"""
inference.py

Unified Inference Engine for Tamil Nadu Crop Yield Prediction & Climate-Risk Classification.
Integrates:
  1. Stage 1 — Crop Yield Regression (Primary Group B with historical lag, Fallback Group A without lag).
  2. Stage 2 — District-Level Climate and Soil Clustering (K-Means k=4).
  3. Stage 3 — Binary Yield-Risk Classification (Training-derived tau_20 threshold, tuned decision threshold).

Strict Quality & Safety Guarantees:
  - Enforces exact ordered feature schemas at inference time.
  - Distinguishes physical range violations (hard errors) from out-of-distribution (OOD) warnings.
  - Seamlessly routes to separate Group A Fallback pipelines when historical yield lag is missing/None,
    explicitly documenting the fallback and warning about lower accuracy. Never imputes 0 or arbitrary means.
  - Returns honest model status flags ('operational', 'research_prototype', 'experimental') and documented caveats.
  - Relies exclusively on genuine serialized model artifacts in model_artifacts/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Cross-platform compatibility shim for unpickling Cython loss modules
if "_loss" not in sys.modules:
    try:
        import sklearn._loss._loss as _loss_mod
        sys.modules["_loss"] = _loss_mod
    except (ImportError, AttributeError):
        try:
            import sklearn._loss as _loss_mod
            sys.modules["_loss"] = _loss_mod
        except (ImportError, AttributeError):
            pass

# Default artifacts directory relative to this file
DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")

# Canonical Crop Mapping
CANONICAL_CROPS = {
    'cotton': 'Cotton(lint)',
    'cotton(lint)': 'Cotton(lint)',
    'cotton_lint': 'Cotton(lint)',
    'groundnut': 'Groundnut',
    'peanut': 'Groundnut',
    'maize': 'Maize',
    'corn': 'Maize',
    'rice': 'Rice',
    'paddy': 'Rice',
    'sugarcane': 'Sugarcane'
}

CROP_FILENAME_MAP = {
    'Cotton(lint)': 'cotton_lint',
    'Groundnut': 'groundnut',
    'Maize': 'maize',
    'Rice': 'rice',
    'Sugarcane': 'sugarcane'
}

GROUP_A_FEATURES = ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K']
GROUP_B_FEATURES = ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K', 'Yield_Lag_Avg']

class InferenceError(Exception):
    """Base exception for inference engine errors."""
    pass

class ValidationError(InferenceError):
    """Raised when user input violates physical limits or expected schema."""
    pass

class ArtifactLoadError(InferenceError):
    """Raised when a required model artifact cannot be loaded."""
    pass

class CropYieldInferenceEngine:
    """
    Production-ready inference engine for Crop Yield Prediction and Climate Risk Assessment.
    """
    def __init__(self, artifacts_dir=DEFAULT_ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self._load_metadata()
        self._load_models()

    def _load_metadata(self):
        meta_file = os.path.join(self.artifacts_dir, "model_metadata.json")
        risk_file = os.path.join(self.artifacts_dir, "risk_thresholds.json")
        clust_file = os.path.join(self.artifacts_dir, "cluster_metadata.json")
        feat_file = os.path.join(self.artifacts_dir, "feature_ranges.json")

        for fpath in [meta_file, risk_file, clust_file, feat_file]:
            if not os.path.exists(fpath):
                raise ArtifactLoadError(f"Required metadata file missing: {fpath}")

        with open(meta_file, "r") as f:
            self.metadata = json.load(f)
        with open(risk_file, "r") as f:
            self.risk_thresholds = json.load(f)
        with open(clust_file, "r") as f:
            self.cluster_metadata = json.load(f)
        with open(feat_file, "r") as f:
            self.feature_ranges = json.load(f)

        self.districts = sorted(list(self.cluster_metadata['district_assignments'].keys()))
        self.districts_upper_map = {d.upper(): d for d in self.districts}

    def _load_models(self):
        self.yield_models_primary = {}
        self.yield_models_fallback = {}
        self.risk_models_primary = {}
        self.risk_models_fallback = {}

        for crop_name in CROP_FILENAME_MAP.keys():
            crop_clean = CROP_FILENAME_MAP[crop_name]

            # Primary regression
            yp_path = os.path.join(self.artifacts_dir, f"yield_model_{crop_clean}.joblib")
            yf_path = os.path.join(self.artifacts_dir, f"yield_model_fallback_{crop_clean}.joblib")
            rp_path = os.path.join(self.artifacts_dir, f"risk_model_{crop_clean}.joblib")
            rf_path = os.path.join(self.artifacts_dir, f"risk_model_fallback_{crop_clean}.joblib")

            for p in [yp_path, yf_path, rp_path, rf_path]:
                if not os.path.exists(p):
                    raise ArtifactLoadError(f"Required model pipeline missing: {p}")

            self.yield_models_primary[crop_name] = joblib.load(yp_path)
            self.yield_models_fallback[crop_name] = joblib.load(yf_path)
            self.risk_models_primary[crop_name] = joblib.load(rp_path)
            self.risk_models_fallback[crop_name] = joblib.load(rf_path)

        # Cluster pipeline
        cl_path = os.path.join(self.artifacts_dir, "cluster_model.joblib")
        if not os.path.exists(cl_path):
            raise ArtifactLoadError(f"Cluster pipeline missing: {cl_path}")
        self.cluster_pipeline = joblib.load(cl_path)

    # --------------------------------------------------------------------------
    # Normalization & Validation Helpers
    # --------------------------------------------------------------------------

    def normalize_crop(self, crop: str) -> str:
        if not isinstance(crop, str):
            raise ValidationError(f"Crop must be a string, got {type(crop).__name__}")
        norm = crop.strip().lower()
        if norm not in CANONICAL_CROPS:
            valid_list = sorted(list(set(CANONICAL_CROPS.values())))
            raise ValidationError(f"Invalid crop '{crop}'. Must be one of: {valid_list}")
        return CANONICAL_CROPS[norm]

    def normalize_district(self, district: str) -> str:
        if not isinstance(district, str):
            raise ValidationError(f"District must be a string, got {type(district).__name__}")
        norm = district.strip().upper()
        if norm not in self.districts_upper_map:
            raise ValidationError(f"Invalid Tamil Nadu district '{district}'. Must be one of: {self.districts}")
        return self.districts_upper_map[norm]

    def validate_inputs(self, rainfall, temp, humidity, ph, n, p, k, yield_lag_avg=None):
        """
        Validates numeric inputs against physical limits and identifies out-of-distribution values.
        Returns a tuple: (cleaned_dict, ood_warnings)
        """
        raw_vals = {
            'Rainfall': rainfall, 'Temp': temp, 'Humidity': humidity,
            'pH': ph, 'N': n, 'P': p, 'K': k
        }
        if yield_lag_avg is not None and not (isinstance(yield_lag_avg, (int, float)) and np.isnan(yield_lag_avg)):
            raw_vals['Yield_Lag_Avg'] = yield_lag_avg

        cleaned = {}
        ood_warnings = []

        feat_specs = self.feature_ranges['features']

        for feat, val in raw_vals.items():
            if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
                if feat == 'Yield_Lag_Avg':
                    continue
                else:
                    raise ValidationError(f"Feature '{feat}' cannot be missing or NaN. Full climate and soil inputs are required.")

            try:
                fval = float(val)
            except (ValueError, TypeError):
                raise ValidationError(f"Feature '{feat}' must be a numeric value, got '{val}'.")

            if np.isinf(fval):
                raise ValidationError(f"Feature '{feat}' cannot be infinite.")

            spec = feat_specs[feat]
            p_min = spec['physical_min']
            p_max = spec['physical_max']

            # Hard physical boundary validation
            if fval < p_min or fval > p_max:
                raise ValidationError(
                    f"Physical limit violation for '{feat}': value {fval} {spec['unit']} "
                    f"is outside valid physical range [{p_min}, {p_max}]."
                )

            # Training distribution percentile check (OOD warning)
            t_p01 = spec['training_p01']
            t_p99 = spec['training_p99']
            if fval < t_p01 or fval > t_p99:
                ood_warnings.append(
                    f"Feature '{feat}' value {fval} {spec['unit']} lies outside the observed "
                    f"training domain [1st percentile: {t_p01}, 99th percentile: {t_p99}]."
                )

            cleaned[feat] = fval

        return cleaned, ood_warnings

    # --------------------------------------------------------------------------
    # Public Inference APIs
    # --------------------------------------------------------------------------

    def predict_yield(self, crop: str, district: str, rainfall: float, temp: float,
                      humidity: float, ph: float, n: float, p: float, k: float,
                      yield_lag_avg: float = None) -> dict:
        """
        Predicts crop yield (in t/ha) using the validated regression pipeline.
        If yield_lag_avg is None, automatically uses the Group A Fallback model.
        """
        crop_canon = self.normalize_crop(crop)
        dist_canon = self.normalize_district(district)
        cleaned_inputs, ood_warnings = self.validate_inputs(
            rainfall, temp, humidity, ph, n, p, k, yield_lag_avg
        )

        has_lag = ('Yield_Lag_Avg' in cleaned_inputs and cleaned_inputs['Yield_Lag_Avg'] > 0)
        crop_meta = self.metadata['crops'][crop_canon]['regression']

        if has_lag:
            feature_group_used = 'Group_B_Lag'
            model = self.yield_models_primary[crop_canon]
            features_ordered = [cleaned_inputs[f] for f in GROUP_B_FEATURES]
            feature_dict = {f: cleaned_inputs[f] for f in GROUP_B_FEATURES}
            model_status = crop_meta['model_status']
            caveats = crop_meta['status_rationale']
        else:
            feature_group_used = 'Group_A_Fallback'
            model = self.yield_models_fallback[crop_canon]
            features_ordered = [cleaned_inputs[f] for f in GROUP_A_FEATURES]
            feature_dict = {f: cleaned_inputs[f] for f in GROUP_A_FEATURES}
            model_status = 'fallback_lower_accuracy'
            caveats = "Using climate-and-soil fallback pipeline (historical yield not provided). Model has lower predictive capacity."

        # Predict
        X_input = np.array([features_ordered], dtype=float)
        raw_pred = model.predict(X_input)[0]
        pred_yield = float(np.clip(raw_pred, 0, None))

        # Check district cluster context
        cluster_info = self.cluster_metadata['district_assignments'][dist_canon]

        return {
            'crop': crop_canon,
            'district': dist_canon,
            'predicted_yield': round(pred_yield, 4),
            'unit': 't/ha',
            'feature_group_used': feature_group_used,
            'features_input': feature_dict,
            'model_family': crop_meta['model_family'],
            'model_status': model_status,
            'is_ood': len(ood_warnings) > 0,
            'ood_warnings': ood_warnings,
            'district_cluster': {
                'cluster_id': cluster_info['cluster_id'],
                'cluster_name': cluster_info['cluster_name'],
                'zone': cluster_info['zone'],
                'soil_confidence': cluster_info['soil_confidence']
            },
            'caveats': caveats,
            'validation_r2': crop_meta['validation_metrics']['R2'],
            'test_r2': crop_meta['test_metrics']['R2']
        }

    def predict_risk(self, crop: str, district: str, rainfall: float, temp: float,
                     humidity: float, ph: float, n: float, p: float, k: float,
                     yield_lag_avg: float = None) -> dict:
        """
        Classifies low-yield risk (High Risk vs Normal) and computes estimated risk probability.
        If yield_lag_avg is None, automatically uses the Group A Fallback classifier.
        """
        crop_canon = self.normalize_crop(crop)
        dist_canon = self.normalize_district(district)
        cleaned_inputs, ood_warnings = self.validate_inputs(
            rainfall, temp, humidity, ph, n, p, k, yield_lag_avg
        )

        has_lag = ('Yield_Lag_Avg' in cleaned_inputs and cleaned_inputs['Yield_Lag_Avg'] > 0)
        crop_meta = self.metadata['crops'][crop_canon]['classification']

        if has_lag:
            feature_group_used = 'Group_B_Lag'
            model = self.risk_models_primary[crop_canon]
            features_ordered = [cleaned_inputs[f] for f in GROUP_B_FEATURES]
            feature_dict = {f: cleaned_inputs[f] for f in GROUP_B_FEATURES}
            decision_th = float(crop_meta['decision_threshold_p'])
            model_status = crop_meta['model_status']
            caveats = crop_meta['status_rationale']
        else:
            feature_group_used = 'Group_A_Fallback'
            model = self.risk_models_fallback[crop_canon]
            features_ordered = [cleaned_inputs[f] for f in GROUP_A_FEATURES]
            feature_dict = {f: cleaned_inputs[f] for f in GROUP_A_FEATURES}
            decision_th = 0.50
            model_status = 'fallback_lower_accuracy'
            caveats = "Using climate-and-soil fallback classifier (historical yield not provided). Lower discrimination."

        X_input = np.array([features_ordered], dtype=float)

        if hasattr(model, "predict_proba"):
            risk_prob = float(model.predict_proba(X_input)[0, 1])
        else:
            risk_prob = float(model.predict(X_input)[0])

        is_high_risk = bool(risk_prob >= decision_th)
        risk_class = "High Risk" if is_high_risk else "Normal"

        cluster_info = self.cluster_metadata['district_assignments'][dist_canon]
        tau20 = float(crop_meta['yield_threshold_tau20'])

        return {
            'crop': crop_canon,
            'district': dist_canon,
            'predicted_risk_class': risk_class,
            'is_high_risk': is_high_risk,
            'risk_probability': round(risk_prob, 4),
            'risk_threshold_tau20_t_ha': round(tau20, 4),
            'decision_threshold_p': round(decision_th, 4),
            'feature_group_used': feature_group_used,
            'features_input': feature_dict,
            'model_family': crop_meta['model_family'],
            'model_status': model_status,
            'is_ood': len(ood_warnings) > 0,
            'ood_warnings': ood_warnings,
            'district_cluster': {
                'cluster_id': cluster_info['cluster_id'],
                'cluster_name': cluster_info['cluster_name'],
                'zone': cluster_info['zone'],
                'soil_confidence': cluster_info['soil_confidence']
            },
            'caveats': caveats,
            'test_roc_auc': crop_meta['test_metrics']['ROC_AUC'],
            'test_recall': crop_meta['test_metrics']['Recall']
        }

    def get_district_profile(self, district: str) -> dict:
        """
        Retrieves Stage 2 clustering profile, geographic coords, and soil confidence for a district.
        """
        dist_canon = self.normalize_district(district)
        d_meta = self.cluster_metadata['district_assignments'][dist_canon]
        cid = str(d_meta['cluster_id'])
        c_profile = self.cluster_metadata['cluster_profiles'][cid]

        return {
            'district': dist_canon,
            'zone': d_meta['zone'],
            'soil_confidence': d_meta['soil_confidence'],
            'latitude': d_meta['latitude'],
            'longitude': d_meta['longitude'],
            'cluster_id': d_meta['cluster_id'],
            'cluster_name': d_meta['cluster_name'],
            'cluster_district_count': c_profile['district_count'],
            'districts_in_cluster': c_profile['districts'],
            'cluster_climate_soil_means': c_profile['climate_soil_means'],
            'district_historical_averages': d_meta['features']
        }

    def list_crops(self) -> list:
        return sorted(list(self.metadata['crops'].keys()))

    def list_districts(self) -> list:
        return self.districts

    def get_feature_schema(self, group='Group_B_Primary') -> dict:
        return self.metadata['feature_schemas'].get(group, {})

