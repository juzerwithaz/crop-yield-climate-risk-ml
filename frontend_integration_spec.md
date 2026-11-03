# Frontend Integration Specification
## Project: District-Level Crop Yield Prediction with Explainable Climate-Risk Clustering (Tamil Nadu)
**Target Framework:** Streamlit (Python-native) with Plotly  
**Inference Engine:** `inference.py` backed by `model_artifacts/`  
**Specification Version:** 1.0 (Production-Ready)  
**Date:** October 2026

---

## 1. System Architecture & Design Philosophy

The upcoming frontend provides an interactive agricultural analytics application for researchers, agronomists, and policy planners in Tamil Nadu. The system architecture avoids unnecessary complexity: because all research pipelines, data transformations, and model artifacts are Python-based, the frontend runs directly on Python using **Streamlit** and **Plotly**.

```
+--------------------------------------------------------------------------+
|                       Streamlit Multi-Page UI                            |
|  [Page 1: Overview]    [Page 2: Yield Predictor] [Page 3: Risk Analyzer] |
|  [Page 4: Agro-Zones]  [Page 5: Explainability]  [Page 6: Methodology]  |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                  CropYieldInferenceEngine (inference.py)                 |
|  + Schema Validation (physical bounds & missing value protection)        |
|  + Out-of-Distribution (OOD) Detector (training percentile check)        |
|  + Dynamic Pipeline Router: Group B (with lag) vs Group A (fallback)    |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                        model_artifacts/ (Disk)                           |
|  + yield_model_<crop>.joblib (5 Primary Regressors)                      |
|  + yield_model_fallback_<crop>.joblib (5 Fallback Regressors)            |
|  + risk_model_<crop>.joblib (5 Primary Classifiers)                      |
|  + risk_model_fallback_<crop>.joblib (5 Fallback Classifiers)            |
|  + cluster_model.joblib (StandardScaler + KMeans k=4)                    |
|  + model_metadata.json, risk_thresholds.json, cluster_metadata.json      |
|  + feature_ranges.json (domain boundaries & [p1, p99] percentiles)       |
+--------------------------------------------------------------------------+
```

### Core Safety Guarantees for UI Rendering
1. **Zero Fabrication:** No hardcoded sample predictions, mocked probabilities, or static charts disguised as live ML output. Every displayed prediction must come from `CropYieldInferenceEngine`.
2. **Honest Model Status Badges:** Every model prediction must display its verified operational status:
   - `Operational`: Validated generalization (e.g. Rice, Groundnut).
   - `Research-Prototype`: Useful benchmark with documented caveats (e.g. Sugarcane regression, Cotton).
   - `Experimental`: Severe unmodeled regime shift or near-random ranking (e.g. Maize regression due to post-2007 surge; Sugarcane classification due to weak ROC-AUC $0.5815$).
3. **Epistemic Labeling:** The UI must visually categorize all displayed information as one of:
   - `[Observed Data]`: Historical records from the master dataset.
   - `[Model Prediction]`: Statistical outputs from trained pipelines.
   - `[Statistical Association]`: Correlation or clustering proximity (not causation).
   - `[Agronomic Hypothesis]`: External domain interpretation requiring field validation.

---

## 2. Page-by-Page Specifications

### Page 1: Overview Dashboard
- **Purpose:** High-level project summary, coverage statistics, and statewide historical trends.
- **Key UI Elements:**
  - **Metric Cards:** 31 Districts, 5 Major Crops, 17 Historical Years (1997–2013), 2,298 Primary Observations (35 invalid zeros cleanly excluded).
  - **Chronological Split Map:** Visual timeline showing Training ($\le 2007$), Validation (2008–2009), and Test (2010–2013).
  - **Statewide Yield Trends:** Plotly time-series chart showing average historical yields for all 5 crops over time, highlighting the dramatic post-2007 surge in Maize.
  - **Data Coverage & Soil Confidence:** Donut chart showing soil data confidence across districts (11 real laboratory surveys, 18 zone averages, 2 low-confidence hill/plateau regions).

### Page 2: Yield Predictor
- **Purpose:** Scenario-based crop yield forecasting with full input validation and fallback routing.
- **Input Controls:**
  - Crop Selector (`Rice`, `Groundnut`, `Cotton(lint)`, `Maize`, `Sugarcane`).
  - District Selector (31 Tamil Nadu districts).
  - Climate Inputs (with physical sliders + number inputs):
    - Annual Rainfall ($\text{mm/year}$, valid: $0\text{--}5000$, training mean: $1,003\text{ mm}$).
    - Mean Temperature ($^\circ\text{C}$, valid: $5\text{--}55$, training mean: $27.0^\circ\text{C}$).
    - Relative Humidity ($\%$, valid: $0\text{--}100$, training mean: $71.1\%$).
  - Soil Inputs (auto-filled with district baseline, editable):
    - Soil pH (valid: $3.0\text{--}11.0$).
    - Available N, P, K ($\text{kg/ha}$, valid: $\ge 0$).
  - Historical Yield Input (`Yield_Lag_Avg` in $\text{t/ha}$):
    - Optional toggle: *"Provide 3-Year Historical Average Yield"*.
    - If toggled OFF or empty: UI calls the Group A Fallback model and displays an informational warning badge.
- **Outputs & Visualizations:**
  - **Predicted Yield Card:** Value in $\text{t/ha}$, model status pill (`Operational`, `Research-Prototype`, `Experimental`).
  - **Out-of-Distribution Alert:** Prominent banner if inputs exceed the 1st or 99th training percentiles.
  - **Historical Comparison Gauge:** Compares predicted yield against the district's historical mean and the statewide 20th–80th percentile range.
  - **Fallback Notice:** If Group A was used, displays: *"Predictions are based on climate and soil features only. Accuracy is lower without historical yield."*

### Page 3: Risk Analyzer
- **Purpose:** Early warning system for low-yield risk with probability threshold interpretation.
- **Input Controls:** Identical input schema to Page 2.
- **Outputs & Visualizations:**
  - **Risk Classification Badge:** `HIGH RISK` (Red) or `NORMAL` (Green).
  - **Probability Dial:** Animated gauge displaying estimated low-yield probability $P(\text{Risk}=1) \in [0, 1]$.
  - **Threshold Explainer:** Visual threshold bar showing:
    - Ground-truth yield threshold $\tau_{20}$ ($\text{t/ha}$ below which a yield is considered low).
    - Probability decision threshold $p^*$ (e.g. $0.45$ for Rice, $0.50$ for Groundnut).
  - **Caveat Banner for Experimental Crops:**
    - For Maize: *"Model has high sampling uncertainty: validation split contained 0 positive instances, and test split had only 2 positive instances."*
    - For Sugarcane: *"Model has weak discrimination (Test ROC-AUC = 0.5815). Low-yield probability should not be used for critical agricultural decisions."*

### Page 4: Agro-Zone Explorer
- **Purpose:** Interactive exploration of Stage 2 environmental clustering.
- **Visual Components:**
  - **Choropleth / Scatter Map:** Tamil Nadu map plotting all 31 districts colored by their Stage 2 Cluster:
    - Cluster 0: Central & Delta Semi-Arid Plains (13 districts).
    - Cluster 1: Western Ghats & Highland Agro-Zone (4 districts).
    - Cluster 2: North-Western Plateau (3 districts).
    - Cluster 3: Coastal Humid Plain (11 districts).
  - **Cluster Profile Inspector:** Selecting a district displays its cluster profile, long-term climate means, soil nutrient breakdown, and data confidence flag.
  - **Radar / Parallel Coordinates Chart:** Multi-feature comparison showing how the selected district's climate and soil deviate from cluster averages.

### Page 5: Model Explainability & Diagnostics
- **Purpose:** Transparent inspection of model mechanics and error diagnostics.
- **Visual Components:**
  - **Feature Importance Chart:** Permutation feature importance for regression and classification models, demonstrating the heavy weighting of `Yield_Lag_Avg` ($45\%\text{--}70\%$) followed by Rainfall and Temperature.
  - **Confusion Matrices & Calibration Plots:** Actual test-set confusion matrices and reliability curves from `ml_audit_results/`.
  - **Residual Distributions:** Error distribution by crop and split from Stage 1.
  - **Post-Hoc Cluster Risk Prevalence:** Table and bar chart displaying historical low-yield risk frequency by agro-climatic cluster.

### Page 6: About & Methodology
- **Purpose:** Academic defense, reproducibility guidelines, and audit documentation.
- **Content:**
  - Data lineage and NASA POWER daily rainfall correction documentation.
  - Explanation of the 35 invalid zero-yield records.
  - Chronological validation design and strict separation of dev and test.
  - Explicit disclosure of prior inspection of the 2010–2013 test set.
  - Instructions for running `test_inference.py` and reproducing all results.

---

## 3. Data & API Contracts

### Input Schema (`CropYieldInferenceEngine.predict_yield`)
```json
{
  "crop": "Rice",
  "district": "Thanjavur",
  "rainfall": 980.5,
  "temp": 27.6,
  "humidity": 72.3,
  "ph": 7.85,
  "n": 250.0,
  "p": 20.0,
  "k": 300.0,
  "yield_lag_avg": 3.42
}
```

### Yield Prediction Output Schema
```json
{
  "crop": "Rice",
  "district": "Thanjavur",
  "predicted_yield": 3.2845,
  "unit": "t/ha",
  "feature_group_used": "Group_B_Lag",
  "model_family": "RandomForest",
  "model_status": "operational",
  "is_ood": false,
  "ood_warnings": [],
  "district_cluster": {
    "cluster_id": 0,
    "cluster_name": "Central & Delta Semi-Arid Plains",
    "zone": "Cauvery Delta",
    "soil_confidence": "real"
  },
  "caveats": "Verified positive test generalization R² with robust historical lag signal.",
  "validation_r2": 0.1575,
  "test_r2": 0.1320
}
```

### Risk Prediction Output Schema
```json
{
  "crop": "Rice",
  "district": "Thanjavur",
  "predicted_risk_class": "Normal",
  "is_high_risk": false,
  "risk_probability": 0.2841,
  "risk_threshold_tau20_t_ha": 2.7434,
  "decision_threshold_p": 0.4500,
  "feature_group_used": "Group_B_Lag",
  "model_family": "RandomForest",
  "model_status": "operational",
  "is_ood": false,
  "ood_warnings": [],
  "district_cluster": {
    "cluster_id": 0,
    "cluster_name": "Central & Delta Semi-Arid Plains",
    "zone": "Cauvery Delta",
    "soil_confidence": "real"
  },
  "caveats": "High discrimination (Test ROC-AUC > 0.90, Recall > 0.80) with consistent historical lag signal.",
  "test_roc_auc": 0.9093,
  "test_recall": 0.8824
}
```

---

## 4. Input Limits & Validation Boundaries

| Feature | Unit | Physical Range [Min, Max] | Training Distribution [p01, p99] | UI Widget |
|---|---|---|---|---|
| **Crop** | categorical | 5 Canonical Crops | Fixed List | Dropdown Select |
| **District** | categorical | 31 TN Districts | Fixed List | Dropdown Select |
| **Rainfall** | $\text{mm/year}$ | $[0.0, 5000.0]$ | $[521.8, 1789.2]$ | Number Input + Slider |
| **Temperature** | $^\circ\text{C}$ | $[5.0, 55.0]$ | $[23.1, 29.8]$ | Number Input + Slider |
| **Relative Humidity** | $\%$ | $[0.0, 100.0]$ | $[62.4, 82.3]$ | Number Input + Slider |
| **Soil pH** | pH scale | $[3.0, 11.0]$ | $[6.09, 8.27]$ | Number Input + Slider |
| **Nitrogen (N)** | $\text{kg/ha}$ | $[0.0, 1000.0]$ | $[140.0, 425.0]$ | Number Input |
| **Phosphorus (P)**| $\text{kg/ha}$ | $[0.0, 300.0]$ | $[13.8, 51.0]$ | Number Input |
| **Potassium (K)** | $\text{kg/ha}$ | $[0.0, 1000.0]$ | $[123.0, 518.0]$ | Number Input |
| **Yield Lag Avg** | $\text{t/ha}$ | $[0.0, 300.0]$ | Crop-dependent | Number Input (Optional) |

---

## 5. UI Error Handling & User Feedback Guidelines

1. **Physical Validation Errors:** Display immediate red inline callouts explaining the physical violation (e.g. *"Rainfall cannot be negative"* or *"Soil pH must be between 3.0 and 11.0"*). Block prediction execution.
2. **Out-of-Distribution Warnings:** Display an amber warning box below the prediction output:  
   *`⚠️ Input Advisory: Annual Rainfall (2,400 mm) exceeds the 99th percentile of historical observations (1,789 mm). Predictions in this regime extrapolate beyond typical weather patterns.`*
3. **Fallback Model Warning:** When historical yield is absent, display a blue informational banner:  
   *`ℹ️ Mode: Fallback Climate-and-Soil Model. Historical 3-year yield was not provided. Model captures macro agro-environmental trends but cannot adjust for farm-specific management or local irrigation.`*
4. **Experimental Crop Warning:** For Maize and Sugarcane, display a persistent caution banner:  
   *`⚠️ Research Advisory: This model is flagged as Experimental. Results reflect research exploration and should not be used for operational decision-making.`*

