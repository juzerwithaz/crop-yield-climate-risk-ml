"""
test_inference.py

Automated Test Suite for CropYieldInferenceEngine & Model Artifacts.
Verifies:
  1. Artifact integrity: all joblib pipelines, metadata JSONs, and clusters load.
  2. Input schema validation: checks physical boundaries, unknown crops/districts.
  3. Prediction validity: non-negative finite yield, risk probabilities in [0, 1].
  4. Decision threshold consistency: binary class strictly matches (probability >= p*).
  5. Fallback routing: correctly falls back to Group A when Yield_Lag_Avg is missing/None.
  6. Out-of-distribution detection: fires non-fatal warnings for extreme inputs.
  7. Non-regression check: asserts existing stage datasets and results remain unaltered.
"""

import os
import unittest
import numpy as np
import pandas as pd
from inference import CropYieldInferenceEngine, ValidationError, ArtifactLoadError

class TestCropYieldInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CropYieldInferenceEngine()
        cls.valid_crops = ['Cotton(lint)', 'Groundnut', 'Maize', 'Rice', 'Sugarcane']
        cls.valid_districts = cls.engine.list_districts()

    # --------------------------------------------------------------------------
    # 1. ARTIFACT & METADATA LOADING
    # --------------------------------------------------------------------------
    def test_all_crops_and_districts_loaded(self):
        self.assertEqual(len(self.valid_districts), 31, "Expected 31 Tamil Nadu districts.")
        crops = self.engine.list_crops()
        self.assertEqual(sorted(crops), sorted(self.valid_crops))

    def test_model_pipelines_exist(self):
        for crop in self.valid_crops:
            self.assertIn(crop, self.engine.yield_models_primary)
            self.assertIn(crop, self.engine.yield_models_fallback)
            self.assertIn(crop, self.engine.risk_models_primary)
            self.assertIn(crop, self.engine.risk_models_fallback)

    # --------------------------------------------------------------------------
    # 2. SCHEMA & VALIDATION TESTS
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
                rainfall=-50.0, temp=28, humidity=70, ph=7.5, n=250, p=20, k=300, yield_lag_avg=3.5
            )

    def test_impossible_ph_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            self.engine.predict_yield(
                crop="Rice", district="Thanjavur",
                rainfall=900, temp=28, humidity=70, ph=14.5, n=250, p=20, k=300, yield_lag_avg=3.5
            )

    def test_missing_climate_feature_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            self.engine.predict_yield(
                crop="Rice", district="Thanjavur",
                rainfall=None, temp=28, humidity=70, ph=7.5, n=250, p=20, k=300
            )

    # --------------------------------------------------------------------------
    # 3. PREDICTION VALIDITY & CONSISTENCY
    # --------------------------------------------------------------------------
    def test_primary_yield_predictions_finite_and_non_negative(self):
        for crop in self.valid_crops:
            res = self.engine.predict_yield(
                crop=crop, district="Thanjavur",
                rainfall=950.0, temp=27.5, humidity=72.0, ph=7.8, n=250.0, p=20.0, k=300.0,
                yield_lag_avg=3.0 if crop != 'Sugarcane' else 100.0
            )
            self.assertIsInstance(res['predicted_yield'], float)
            self.assertGreaterEqual(res['predicted_yield'], 0.0)
            self.assertEqual(res['feature_group_used'], 'Group_B_Lag')
            self.assertEqual(res['unit'], 't/ha')

    def test_fallback_routing_when_lag_is_none(self):
        res = self.engine.predict_yield(
            crop="Rice", district="Madurai",
            rainfall=850.0, temp=28.0, humidity=68.0, ph=7.6, n=205.0, p=16.0, k=300.0,
            yield_lag_avg=None
        )
        self.assertEqual(res['feature_group_used'], 'Group_A_Fallback')
        self.assertEqual(res['model_status'], 'fallback_lower_accuracy')
        self.assertIn("fallback", res['caveats'].lower())
        self.assertGreaterEqual(res['predicted_yield'], 0.0)

    def test_risk_prediction_probability_and_threshold_consistency(self):
        for crop in self.valid_crops:
            res = self.engine.predict_risk(
                crop=crop, district="Coimbatore",
                rainfall=1050.0, temp=24.0, humidity=74.0, ph=7.8, n=330.0, p=30.0, k=340.0,
                yield_lag_avg=2.5 if crop != 'Sugarcane' else 95.0
            )
            prob = res['risk_probability']
            th = res['decision_threshold_p']
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            if prob >= th:
                self.assertEqual(res['predicted_risk_class'], "High Risk")
                self.assertTrue(res['is_high_risk'])
            else:
                self.assertEqual(res['predicted_risk_class'], "Normal")
                self.assertFalse(res['is_high_risk'])

    def test_out_of_distribution_warning(self):
        # Rainfall of 3500 mm is physically possible (<5000), but > 99th percentile (training p99 ~ 1700)
        res = self.engine.predict_yield(
            crop="Rice", district="Thanjavur",
            rainfall=3500.0, temp=27.5, humidity=72.0, ph=7.8, n=250.0, p=20.0, k=300.0,
            yield_lag_avg=3.5
        )
        self.assertTrue(res['is_ood'])
        self.assertTrue(any('Rainfall' in w for w in res['ood_warnings']))

    # --------------------------------------------------------------------------
    # 4. DISTRICT CLUSTERING PROFILE API
    # --------------------------------------------------------------------------
    def test_district_clustering_profile(self):
        prof = self.engine.get_district_profile("The Nilgiris")
        self.assertEqual(prof['district'], "THE NILGIRIS")
        self.assertEqual(prof['cluster_id'], 1)
        self.assertEqual(prof['cluster_name'], "Western Ghats & Highland Agro-Zone")
        self.assertEqual(prof['soil_confidence'], "low-confidence")
        self.assertIn("COIMBATORE", prof['districts_in_cluster'])

    # --------------------------------------------------------------------------
    # 5. PRESERVATION OF EXISTING STAGE FILES (NON-REGRESSION)
    # --------------------------------------------------------------------------
    def test_existing_stage_files_unaltered(self):
        # 1. Master dataset
        self.assertTrue(os.path.exists("tn_master_dataset_stage1_ready.csv"))
        df_master = pd.read_csv("tn_master_dataset_stage1_ready.csv")
        self.assertEqual(df_master.shape, (2333, 22), "Master dataset shape altered!")
        
        # 2. Stage 1 results
        self.assertTrue(os.path.exists("stage1_results/stage1_test_metrics.csv"))
        
        # 3. Stage 2 results
        self.assertTrue(os.path.exists("stage2_results/stage2_cluster_assignments.csv"))
        df_clust = pd.read_csv("stage2_results/stage2_cluster_assignments.csv")
        self.assertEqual(len(df_clust), 31, "Cluster assignments row count altered!")
        
        # 4. Stage 3 results
        self.assertTrue(os.path.exists("stage3_results/stage3_test_metrics.csv"))

if __name__ == '__main__':
    unittest.main()

