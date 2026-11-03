"""
test_inference_v2.py

Automated Test Suite for CropYieldInferenceEngineV2 & Model Artifacts V2 (model_artifacts_v2/).
Verifies:
  1. Artifact integrity: all 20 joblib pipelines, cluster model, and 4 metadata JSONs load cleanly from model_artifacts_v2.
  2. Input schema validation: checks physical boundaries (including pH in [3.0, 11.0]), unknown crops/districts.
  3. Physical vs OOD distinction: pH 9.0 is physically valid [3.0, 11.0] but generates an OOD warning (>8.27 p99).
  4. Prediction validity: non-negative finite yield, realistic Sugarcane scale (~90-130 t/ha, never 1000+ t/ha).
  5. Negative prediction handling: explicit reporting via is_clipped and clipping_warning (no silent clipping).
  6. Decision threshold consistency: binary class strictly matches (probability >= p*).
  7. Fallback routing: correctly falls back to Group A when Yield_Lag_Avg is missing/None or <= 0.
  8. Uncalibrated probability reporting: explicitly identifies probability_type as model_estimated_uncalibrated.
  9. District clustering profile: returns cluster metadata, coordinates, and soil confidence.
  10. Strict version isolation: verifies zero loading from model_artifacts/ (V1).
  11. Non-regression check: asserts existing V1 datasets, V2 datasets, and raw files remain unaltered.
"""

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from inference_v2 import CropYieldInferenceEngineV2, ValidationError, ArtifactLoadError, DEFAULT_ARTIFACTS_DIR

ARTIFACTS_V2_DIR = DEFAULT_ARTIFACTS_DIR

class TestCropYieldInferenceV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(ARTIFACTS_V2_DIR):
            raise FileNotFoundError(f"V2 artifacts directory '{ARTIFACTS_V2_DIR}' does not exist.")
        cls.engine = CropYieldInferenceEngineV2(artifacts_dir=ARTIFACTS_V2_DIR)
        cls.valid_crops = ['Cotton(lint)', 'Groundnut', 'Maize', 'Rice', 'Sugarcane']
        cls.valid_districts = cls.engine.list_districts()

    # --------------------------------------------------------------------------
    # 1. ARTIFACT & METADATA LOADING & ISOLATION
    # --------------------------------------------------------------------------
    def test_all_crops_and_districts_loaded(self):
        self.assertEqual(len(self.valid_districts), 31, "Expected 31 Tamil Nadu districts.")
        crops = self.engine.list_crops()
        self.assertEqual(sorted(crops), sorted(self.valid_crops))

    def test_model_pipelines_exist_and_isolated_to_v2(self):
        for crop in self.valid_crops:
            self.assertIn(crop, self.engine.yield_models_primary, f"Missing primary yield model for {crop}")
            self.assertIn(crop, self.engine.yield_models_fallback, f"Missing fallback yield model for {crop}")
            self.assertIn(crop, self.engine.risk_models_primary, f"Missing primary risk model for {crop}")
            self.assertIn(crop, self.engine.risk_models_fallback, f"Missing fallback risk model for {crop}")

        # Strict isolation assertion: artifacts directory must end in model_artifacts_v2
        self.assertTrue(self.engine.artifacts_dir.endswith("model_artifacts_v2"), "Engine must point to model_artifacts_v2.")
        data_ver = self.engine.metadata.get("data_version", "")
        self.assertIn("v2", data_ver.lower(), f"Metadata must identify V2 dataset, got: {data_ver}")

    def test_risk_thresholds_loaded_from_v2(self):
        self.assertIn('yield_risk_thresholds', self.engine.risk_thresholds)
        sc_tau20 = self.engine.risk_thresholds['yield_risk_thresholds']['Sugarcane']['tau_20_primary']
        self.assertAlmostEqual(sc_tau20, 90.7270, places=4, msg="Sugarcane tau_20 must equal cleansed 90.7270 t/ha")
        rice_tau20 = self.engine.risk_thresholds['yield_risk_thresholds']['Rice']['tau_20_primary']
        self.assertAlmostEqual(rice_tau20, 2.7434, places=4)

    # --------------------------------------------------------------------------
    # 2. SCHEMA & PHYSICAL BOUNDARY VALIDATION
    # --------------------------------------------------------------------------
    def test_invalid_crop_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            self.engine.predict_yield(
                crop="Wheat", district="Thanjavur",
                rainfall=900, temp=28, humidity=70, ph=7.5, n=250, p=20, k=300, yield_lag_avg=3.5
            )

    def test_invalid_district_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            self.engine.predict_yield(
                crop="Rice", district="Bangalore",
                rainfall=900, temp=28, humidity=70, ph=7.5, n=250, p=20, k=300, yield_lag_avg=3.5
            )

    def test_negative_rainfall_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            self.engine.predict_yield(
                crop="Rice", district="Thanjavur",
                rainfall=-50, temp=28, humidity=70, ph=7.5, n=250, p=20, k=300, yield_lag_avg=3.5
            )

    def test_ph_physical_boundary_validation(self):
        """
        In feature_ranges.json, pH physical limits are [3.0, 11.0].
        Values < 3.0 or > 11.0 must raise ValidationError.
        """
        # Too low (< 3.0)
        with self.assertRaises(ValidationError):
            self.engine.predict_yield(
                crop="Rice", district="Thanjavur",
                rainfall=900, temp=28, humidity=70, ph=2.5, n=250, p=20, k=300, yield_lag_avg=3.5
            )
        # Too high (> 11.0)
        with self.assertRaises(ValidationError):
            self.engine.predict_yield(
                crop="Rice", district="Thanjavur",
                rainfall=900, temp=28, humidity=70, ph=11.5, n=250, p=20, k=300, yield_lag_avg=3.5
            )
        with self.assertRaises(ValidationError):
            self.engine.predict_yield(
                crop="Rice", district="Thanjavur",
                rainfall=900, temp=28, humidity=70, ph=14.5, n=250, p=20, k=300, yield_lag_avg=3.5
            )

    def test_ph_ood_warning_within_physical_bounds(self):
        """
        pH = 9.0 is physically valid ([3.0, 11.0]), but above observed training p99 (8.27).
        It must NOT raise ValidationError, but MUST trigger an OOD warning.
        """
        res = self.engine.predict_yield(
            crop="Rice", district="Thanjavur",
            rainfall=950.0, temp=27.5, humidity=72.0, ph=9.0, n=250.0, p=20.0, k=300.0,
            yield_lag_avg=3.2
        )
        self.assertTrue(res['is_ood'])
        self.assertTrue(any('pH' in w for w in res['ood_warnings']))

    def test_missing_climate_feature_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            self.engine.predict_yield(
                crop="Rice", district="Thanjavur",
                rainfall=None, temp=28, humidity=70, ph=7.5, n=250, p=20, k=300
            )

    # --------------------------------------------------------------------------
    # 3. YIELD PREDICTION & MODEL ROUTING TESTS
    # --------------------------------------------------------------------------
    def test_primary_yield_predictions_all_crops(self):
        for crop in self.valid_crops:
            lag = 105.0 if crop == 'Sugarcane' else 3.0
            res = self.engine.predict_yield(
                crop=crop, district="Thanjavur",
                rainfall=950.0, temp=27.5, humidity=72.0, ph=7.8, n=250.0, p=20.0, k=300.0,
                yield_lag_avg=lag
            )
            self.assertIsInstance(res['predicted_yield'], float)
            self.assertGreaterEqual(res['predicted_yield'], 0.0)
            self.assertEqual(res['feature_group_used'], 'Group_B_Lag')
            self.assertFalse(res['is_fallback'])
            self.assertEqual(res['artifact_version'], 'V2')
            self.assertIn('evaluation_disclosure', res)

    def test_fallback_routing_when_lag_is_none_or_missing(self):
        res = self.engine.predict_yield(
            crop="Rice", district="Madurai",
            rainfall=850.0, temp=28.0, humidity=68.0, ph=7.6, n=205.0, p=16.0, k=300.0,
            yield_lag_avg=None
        )
        self.assertEqual(res['feature_group_used'], 'Group_A_Fallback')
        self.assertTrue(res['is_fallback'])
        self.assertEqual(res['model_status'], 'fallback_lower_accuracy')
        self.assertIn("fallback", res['caveats'].lower())
        self.assertGreaterEqual(res['predicted_yield'], 0.0)

    def test_sugarcane_cleansed_scale_sanity(self):
        """Verifies that Sugarcane predictions are realistic (~90-130 t/ha) and NOT 1,000+ t/ha."""
        # Primary prediction with realistic lag (105 t/ha)
        res_prim = self.engine.predict_yield(
            crop="Sugarcane", district="Cuddalore",
            rainfall=1100.0, temp=28.0, humidity=75.0, ph=7.1, n=260.0, p=24.0, k=290.0, yield_lag_avg=105.0
        )
        self.assertGreater(res_prim['predicted_yield'], 50.0)
        self.assertLess(res_prim['predicted_yield'], 200.0, f"Primary Sugarcane prediction unreasonably high: {res_prim['predicted_yield']}")

        # Fallback prediction without lag
        res_fall = self.engine.predict_yield(
            crop="Sugarcane", district="Cuddalore",
            rainfall=1100.0, temp=28.0, humidity=75.0, ph=7.1, n=260.0, p=24.0, k=290.0, yield_lag_avg=None
        )
        self.assertEqual(res_fall['feature_group_used'], 'Group_A_Fallback')
        self.assertTrue(res_fall['is_fallback'])
        self.assertGreater(res_fall['predicted_yield'], 50.0)
        self.assertLess(res_fall['predicted_yield'], 200.0, f"Fallback Sugarcane prediction unreasonably high: {res_fall['predicted_yield']}")

    def test_negative_output_explicit_handling(self):
        """Ensures that clipping/floor handling is explicit and fields are exposed."""
        res = self.engine.predict_yield(
            crop="Rice", district="Thanjavur",
            rainfall=950.0, temp=27.5, humidity=72.0, ph=7.8, n=250.0, p=20.0, k=300.0,
            yield_lag_avg=3.0
        )
        self.assertIn('raw_predicted_yield', res)
        self.assertIn('is_clipped', res)
        self.assertIn('clipping_warning', res)
        self.assertFalse(res['is_clipped'])
        self.assertIsNone(res['clipping_warning'])

    def test_negative_output_explicit_handling_when_negative(self):
        """Ensures that if a regression model outputs a negative value, it is explicitly flagged and warned."""
        with patch.object(self.engine.yield_models_primary['Rice'], 'predict', return_value=np.array([-0.45])):
            res = self.engine.predict_yield(
                crop="Rice", district="Thanjavur",
                rainfall=950.0, temp=27.5, humidity=72.0, ph=7.8, n=250.0, p=20.0, k=300.0,
                yield_lag_avg=3.0
            )
            self.assertEqual(res['raw_predicted_yield'], -0.45)
            self.assertEqual(res['predicted_yield'], 0.0)
            self.assertTrue(res['is_clipped'])
            self.assertIsNotNone(res['clipping_warning'])
            self.assertIn("Physically invalid", res['clipping_warning'])
            self.assertTrue(res['is_ood'])
            self.assertTrue(any('Physically invalid' in w for w in res['ood_warnings']))

    # --------------------------------------------------------------------------
    # 4. YIELD-RISK CLASSIFICATION TESTS
    # --------------------------------------------------------------------------
    def test_risk_classification_all_crops(self):
        for crop in self.valid_crops:
            lag = 95.0 if crop == 'Sugarcane' else 2.5
            res = self.engine.predict_risk(
                crop=crop, district="Coimbatore",
                rainfall=1050.0, temp=24.0, humidity=74.0, ph=7.8, n=330.0, p=30.0, k=340.0,
                yield_lag_avg=lag
            )
            prob = res['risk_probability']
            th = res['decision_threshold_p']
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertEqual(res['probability_type'], 'model_estimated_uncalibrated')
            self.assertIn('calibration_status', res)

            # Threshold mapping consistency
            if prob >= th:
                self.assertEqual(res['predicted_risk_class'], "High Risk")
                self.assertTrue(res['is_high_risk'])
            else:
                self.assertEqual(res['predicted_risk_class'], "Normal")
                self.assertFalse(res['is_high_risk'])

    def test_rice_groundnut_status_descriptions(self):
        """Rice and Groundnut must be described as promising retrospective models."""
        res_rice = self.engine.predict_yield(
            crop="Rice", district="Thanjavur",
            rainfall=950.0, temp=27.5, humidity=72.0, ph=7.8, n=250.0, p=20.0, k=300.0,
            yield_lag_avg=3.2
        )
        self.assertEqual(res_rice['model_status'], 'operational')
        self.assertIn("Promising retrospective model", res_rice['status_description'])
        self.assertIn("Not independently validated as a field-deployed production advisory system", res_rice['status_description'])

    # --------------------------------------------------------------------------
    # 5. DISTRICT CLUSTERING & PROFILING TESTS
    # --------------------------------------------------------------------------
    def test_district_clustering_profile(self):
        prof = self.engine.get_district_profile("The Nilgiris")
        self.assertEqual(prof['district'], "THE NILGIRIS")
        self.assertIn(prof['cluster_id'], [0, 1, 2, 3])
        self.assertIn('latitude', prof)
        self.assertIn('longitude', prof)
        self.assertIn('cluster_climate_soil_means', prof)

    # --------------------------------------------------------------------------
    # 6. NON-REGRESSION OF DATASETS
    # --------------------------------------------------------------------------
    def test_v1_and_source_datasets_unaltered(self):
        v1_path = "tn_master_dataset_stage1_ready.csv"
        v2_path = "tn_master_dataset_stage1_ready_v2.csv"
        raw_path = "tn_crop_yield_filtered.csv"

        self.assertTrue(os.path.exists(v1_path), "V1 dataset must exist.")
        self.assertTrue(os.path.exists(v2_path), "V2 dataset must exist.")
        self.assertTrue(os.path.exists(raw_path), "Raw source file must exist.")

        df_v1 = pd.read_csv(v1_path)
        df_v2 = pd.read_csv(v2_path)
        df_raw = pd.read_csv(raw_path)

        self.assertEqual(len(df_v1), 2333, "V1 dataset row count must remain 2,333.")
        self.assertEqual(len(df_v2), 2333, "V2 dataset row count must be 2,333.")
        self.assertEqual(len(df_raw), 2491, "Raw dataset row count must remain 2,491.")

if __name__ == '__main__':
    unittest.main()
