"""
test_frontend_integration.py
Automated Integration and Verification Test for Tamil Nadu Crop Intelligence Frontend.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

# Ensure root directory is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.engine_provider import get_inference_engine
from utils.data_loader import (
    load_v2_dataset,
    load_regression_comparison,
    load_classification_comparison,
    load_risk_thresholds_comparison,
    extract_v2_model_weights_summary,
    load_v1_feature_importance_benchmark,
    get_dataset_summary_stats
)
from ui_pages.overview import render_overview_page
from ui_pages.yield_predictor import render_yield_predictor_page
from ui_pages.risk_analyzer import render_risk_analyzer_page
from ui_pages.agro_zones import render_agro_zones_page
from ui_pages.explainability import render_explainability_page
from ui_pages.methodology import render_methodology_page

class TestFrontendIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = get_inference_engine()

    def test_01_data_loaders(self):
        """Verify all cached data loaders execute and return non-empty dataframes."""
        df_v2 = load_v2_dataset()
        self.assertEqual(len(df_v2), 2333, "V2 dataset must have exactly 2,333 records")

        stats = get_dataset_summary_stats()
        self.assertEqual(stats['total_records'], 2333)
        self.assertEqual(stats['eligible_records'], 2293)
        self.assertEqual(stats['total_excluded'], 40)
        self.assertEqual(stats['districts_count'], 31)
        self.assertEqual(stats['crops_count'], 5)

        reg_comp = load_regression_comparison()
        self.assertGreaterEqual(len(reg_comp), 10)

        cls_comp = load_classification_comparison()
        self.assertGreaterEqual(len(cls_comp), 10)

        weights = extract_v2_model_weights_summary()
        self.assertIn('Rice_regression', weights)
        self.assertIn('Rice_classification', weights)

        v1_imp = load_v1_feature_importance_benchmark()
        self.assertFalse(v1_imp.empty)
        self.assertIn('Provenance', v1_imp.columns)

    def test_02_all_crops_yield_prediction_primary(self):
        """Verify primary yield prediction works across all 5 crops."""
        crops = ['Cotton(lint)', 'Groundnut', 'Maize', 'Rice', 'Sugarcane']
        district = 'THANJAVUR'
        lags = {'Cotton(lint)': 1.5, 'Groundnut': 1.8, 'Maize': 2.5, 'Rice': 3.5, 'Sugarcane': 105.0}

        for crop in crops:
            res = self.engine.predict_yield(
                crop=crop,
                district=district,
                rainfall=950.0,
                temp=28.0,
                humidity=72.0,
                ph=7.5,
                n=250.0,
                p=25.0,
                k=300.0,
                yield_lag_avg=lags[crop]
            )
            self.assertEqual(res['crop'], crop)
            self.assertEqual(res['feature_group_used'], 'Group_B_Lag')
            self.assertFalse(res['is_fallback'])
            self.assertGreater(res['predicted_yield'], 0.0)
            self.assertIn(res['model_status'], ['operational', 'research_prototype', 'experimental'])

    def test_03_all_crops_yield_prediction_fallback(self):
        """Verify fallback yield prediction routes properly when lag is None."""
        crops = ['Cotton(lint)', 'Groundnut', 'Maize', 'Rice', 'Sugarcane']
        district = 'COIMBATORE'

        for crop in crops:
            res = self.engine.predict_yield(
                crop=crop,
                district=district,
                rainfall=1050.0,
                temp=26.0,
                humidity=70.0,
                ph=6.8,
                n=220.0,
                p=20.0,
                k=280.0,
                yield_lag_avg=None
            )
            self.assertEqual(res['crop'], crop)
            self.assertEqual(res['feature_group_used'], 'Group_A_Fallback')
            self.assertTrue(res['is_fallback'])
            self.assertGreaterEqual(res['predicted_yield'], 0.0)

    def test_04_all_crops_risk_prediction(self):
        """Verify low-yield risk classification across all 5 crops."""
        crops = ['Cotton(lint)', 'Groundnut', 'Maize', 'Rice', 'Sugarcane']
        district = 'MADURAI'

        for crop in crops:
            res = self.engine.predict_risk(
                crop=crop,
                district=district,
                rainfall=850.0,
                temp=29.0,
                humidity=65.0,
                ph=7.8,
                n=240.0,
                p=22.0,
                k=270.0,
                yield_lag_avg=None
            )
            self.assertEqual(res['crop'], crop)
            self.assertIn(res['predicted_risk_class'], ['High Risk', 'Normal'])
            self.assertIn(res['probability_type'], ['model_estimated_uncalibrated'])
            self.assertGreaterEqual(res['risk_probability'], 0.0)
            self.assertLessEqual(res['risk_probability'], 1.0)
            self.assertGreater(res['risk_threshold_tau20_t_ha'], 0.0)
            self.assertGreater(res['decision_threshold_p'], 0.0)

    def test_05_district_profile_and_clusters(self):
        """Verify district profile retrieval across all 4 clusters."""
        districts = ['THANJAVUR', 'COIMBATORE', 'SALEM', 'CUDDALORE']
        for d in districts:
            prof = self.engine.get_district_profile(d)
            self.assertEqual(prof['district'], d)
            self.assertIn(prof['cluster_id'], [0, 1, 2, 3])
            self.assertIn(prof['soil_confidence'], ['real', 'zone-avg', 'low-confidence'])
            self.assertGreater(prof['latitude'], 7.0)
            self.assertGreater(prof['longitude'], 75.0)

    def test_06_page_render_smoke_test(self):
        """Verify that all six page render functions execute without Python exceptions."""
        try:
            render_overview_page(self.engine)
            render_yield_predictor_page(self.engine)
            render_risk_analyzer_page(self.engine)
            render_agro_zones_page(self.engine)
            render_explainability_page(self.engine)
            render_methodology_page()
        except Exception as e:
            self.fail(f"Page rendering raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()

