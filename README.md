# District-Level Crop Yield Prediction with Explainable Climate-Risk Clustering
### *Tamil Nadu Crop Intelligence: Agricultural Machine Learning & Risk Analytics Platform*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Model%20Status-V2%20Cleansed-brightgreen.svg)]()

---

## 📌 Executive Summary

**Tamil Nadu Crop Intelligence** is an end-to-end applied machine learning platform designed to forecast agricultural yields, model agro-climatic typologies, and provide early warnings for extreme low-yield risk across **31 administrative districts** in the state of Tamil Nadu, India.

Spanning **17 historical crop years (1997–2013)** and covering **5 major agricultural crops** (Rice, Groundnut, Maize, Cotton, and Sugarcane), the project integrates satellite-derived meteorological reanalysis, official soil fertility surveys, and rigorous multi-source data provenance auditing into a reproducible, production-grade analytics system.

```
+--------------------------------------------------------------------------------------------------+
|                                    TAMIL NADU CROP INTELLIGENCE                                  |
|                                                                                                  |
|   [Stage 1: Cleansing & Lags]   -->   [Stage 2: Agro-Clustering]   -->   [Stage 3: Risk Class.]  |
|   - NASA POWER aggregation            - K-Means (k=4) Zones               - Binary tau_20 cutoff |
|   - 1997 Sugarcane 0.10x unit         - Coordinate marker mapping         - Tuned decision (p*)  |
|   - 2008 Bifurcation exclusion        - Soil survey confidence            - Uncalibrated notice  |
|   - 35 raw NaN corrections                                                                       |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                            CropYieldInferenceEngineV2 (inference_v2.py)                          |
|   - Dynamic Routing: Group B Primary (with lag) vs Group A Fallback (climate & soil only)        |
|   - Physical Range Validation & Out-of-Distribution (OOD) Percentile Alerts                      |
|   - Explicit non-silent physical floor handling for negative regression outputs                   |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                               Streamlit Analytics Dashboard (app.py)                             |
|   [Overview Dashboard]  [Yield Predictor]  [Risk Analyzer]  [Agro-Zones]  [Explainability]       |
+--------------------------------------------------------------------------------------------------+
```

---

## 🌾 Scope & Agricultural Coverage

The system evaluates five distinct agronomic regimes across Tamil Nadu:

1. **Rice (Paddy):** Highly irrigated, primary foodgrain staple concentrated in the Cauvery Delta.
2. **Groundnut (Peanut):** Semi-arid oilseed cultivated across dryland and transitional tracts.
3. **Maize (Corn):** Rapidly expanding feed cereal characterized by a major post-2007 hybrid seed expansion.
4. **Cotton(lint):** Highly volatile commercial fiber crop predominantly grown under rainfed conditions.
5. **Sugarcane:** High-input, 10–14 month perennial commercial crop sensitive to irrigation water availability.

---

## 🔬 Core Methodological Stages

### Stage 1: Data Cleansing, Provenance & Lag Engineering
The authoritative **V2 Cleansed Dataset** (`tn_master_dataset_stage1_ready_v2.csv`, 2,333 records; 2,293 eligible primary observations, 40 cleanly excluded records) addressed four critical historical data anomalies:

1. **1997 Sugarcane Production Unit Scaling:** Multi-source provenance tracing confirmed a 10x reporting unit error in the 1997 Sugarcane statistics. Production was scaled by **$0.10\times$** (adjusting area-weighted yield from an unphysical $1,085.5\text{ t/ha}$ down to $108.55\text{ t/ha}$), restoring biological plausibility while preserving raw values in provenance lineage fields.
2. **2008 Coimbatore–Tiruppur Administrative Bifurcation:** During the 2008–2009 formation of Tiruppur district, Coimbatore reported undivided aggregate production against post-bifurcation land area, producing artificial spikes (e.g. $25.68\text{ t/ha}$ in Maize). All 5 Coimbatore 2008 crop records are flagged as `EXCLUDED_ADMIN_BIFURCATION` and omitted from model training.
3. **Correction of 35 Missing-Production Artificial Zeros:** Lineage audits traced 35 records with $0.0\text{ t/ha}$ yields back to raw state returns where production was missing (`NaN`). Prior aggregation scripts had inadvertently converted `NaN` to `0.0`. These were flagged as `EXCLUDED_SOURCE_MISSING_PRODUCTION` and restored to missing in active fields.
4. **Leak-Free Calendar Lag Engineering:** Historical rolling 3-year averages (`Yield_Lag_Avg`) were regenerated strictly using prior calendar crop years ($t-1, t-2, t-3$), completely preventing leakage from concurrent harvest periods or administrative exclusions.
5. **Meteorological Feature Alignment:** Gridded daily weather data from NASA POWER was aggregated to cumulative seasonal and annual precipitation totals and annual temperature/humidity means. *(Clarification: This represents precipitation aggregation and unit conversion/alignment, not a ground-station bias correction).*

---

### Stage 2: Agro-Climatic & Soil Typology (K-Means Clustering)
Unsupervised K-Means clustering ($k=4$) partitioned Tamil Nadu's 31 districts into coherent agro-environmental zones using normalized climate (rainfall, temperature, humidity) and soil fertility features (pH, N, P, K):

- **Cluster 0: Central & Delta Semi-Arid Plains (13 districts):** Cauvery Delta and semi-arid central plains; major rice and sugarcane belt.
- **Cluster 1: Western Ghats & Highland Agro-Zone (4 districts):** Elevated, high-rainfall, moderate-temperature western districts (Coimbatore, Krishnagiri, The Nilgiris, Tiruppur).
- **Cluster 2: North-Western Plateau (3 districts):** Inland plateau with moderate precipitation (Dharmapuri, Namakkal, Salem).
- **Cluster 3: Coastal Humid Plain (11 districts):** Coastal maritime boundary with elevated humidity and variable northeast monsoon precipitation.
- **Soil Survey Confidence Accounting:** Dynamically derives and reports data confidence: 11 districts with direct laboratory surveys (`real`), 18 districts with agro-climatic zone-average estimates (`zone-avg`), and 2 low-confidence hill reference regions.

---

### Stage 3: Binary Low-Yield Risk Classification
To warn against severe agricultural downturns, low-yield risk is modeled as a binary classification problem:
- **Ground-Truth Target Cutoff ($\tau_{20}$):** Defined strictly as the **crop-specific 20th percentile of eligible training yields** ($\le 2007$). Observations with yields $\le \tau_{20}$ are labeled `High Risk`.
- **Decision Probability Cutoff ($p^*$):** Tuned on the independent validation split ($2008\text{--}2009$) to optimize the precision-recall trade-off (e.g., $p^* = 0.45$ for Rice, $0.55$ for Sugarcane, $0.50$ default).
- **Uncalibrated Probability Disclosure:** All probability scores are model-estimated tree ensemble / logistic outputs and are explicitly disclosed as uncalibrated via Platt scaling or isotonic regression.

---

### Stage 4: Production Inference Engine & Streamlit Dashboard
The V2 inference pipeline is unified inside [`inference_v2.py`](file:///c:/Users/Admin/Desktop/Crop_Prod/inference_v2.py):
- **Dynamic Routing:** Automatically routes to **Group B Primary** when historical 3-year yield lag is provided, or gracefully falls back to **Group A Fallback** (climate and soil only) with explicit advisory warnings.
- **Physical Boundary Validation:** Enforces strict physical domains (Rainfall $[0, 5000\text{ mm}]$, Temp $[5, 55^\circ\text{C}]$, pH $[3.0, 11.0]$, Nutrients $\ge 0$).
- **Out-of-Distribution (OOD) Detector:** Warns whenever user inputs fall outside the observed training distribution ($< 1\text{st}$ or $> 99\text{th}$ percentile).
- **Non-Silent Physical Floor Handling:** If an extreme input causes a linear model to produce a negative yield, the engine clamps to $0.0\text{ t/ha}$, records the raw output, and surfaces an explicit clipping warning.

---

## 📊 Model Performance Comparison: V1 Baseline vs V2 Cleansed

All models were evaluated on the **protected test split (2010–2013)**. Excluded records were strictly omitted:

### 1. Regression Generalization ($R^2$, RMSE, MAE)

| Crop | Pipeline | Model Family | V1 Test $R^2$ | V2 Test $R^2$ | $\Delta R^2$ | V1 RMSE | V2 RMSE | $\Delta$ RMSE | Operational Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cotton(lint)** | Primary (Lag) | Ridge | $-0.1802$ | **$-0.2394$** | $-0.0592$ | $0.838\text{ t/ha}$ | $0.858\text{ t/ha}$ | $+0.020$ | **Research Prototype** |
| Cotton(lint) | Fallback | Ridge | $-0.5171$ | **$-0.5822$** | $-0.0651$ | $0.950\text{ t/ha}$ | $0.970\text{ t/ha}$ | $+0.020$ | Fallback (Lower Acc) |
| **Groundnut** | Primary (Lag) | HistGradBoost | $0.1674$ | **$0.2103$** | **$+0.0429$** | $0.857\text{ t/ha}$ | $0.834\text{ t/ha}$ | **$-0.023$** | **Operational (Promising)** |
| Groundnut | Fallback | HistGradBoost | $-0.5345$ | **$-0.5324$** | $+0.0021$ | $1.163\text{ t/ha}$ | $1.162\text{ t/ha}$ | $-0.001$ | Fallback (Lower Acc) |
| **Maize** | Primary (Lag) | GradBoost | $-1.4992$ | **$+0.0568$** | **$+1.5560$** | $2.647\text{ t/ha}$ | $1.626\text{ t/ha}$ | **$-1.021$** | **Experimental** |
| Maize | Fallback | GradBoost | $-1.2547$ | **$-1.2316$** | $+0.0231$ | $2.514\text{ t/ha}$ | $2.501\text{ t/ha}$ | $-0.013$ | Fallback (Lower Acc) |
| **Rice** | Primary (Lag) | RandomForest | $0.1320$ | **$0.1963$** | **$+0.0643$** | $1.043\text{ t/ha}$ | $1.003\text{ t/ha}$ | **$-0.040$** | **Operational (Promising)** |
| Rice | Fallback | RandomForest | $-0.1741$ | **$-0.2129$** | $-0.0389$ | $1.213\text{ t/ha}$ | $1.233\text{ t/ha}$ | $+0.020$ | Fallback (Lower Acc) |
| **Sugarcane** | Primary (Lag) | RandomForest | $0.0598$ | **$0.0848$** | **$+0.0250$** | $15.26\text{ t/ha}$ | $15.05\text{ t/ha}$ | **$-0.210$** | **Research Prototype** |
| Sugarcane | Fallback | RandomForest | $-24.5192$ | **$+0.0511$** | **$+24.5703$** | $79.19\text{ t/ha}$ | $15.27\text{ t/ha}$ | **$-63.92$** | Fallback (Restored) |

---

### 2. Risk Classification Generalization (Test: 2010–2013)

| Crop | Model Family | Decision $p^*$ | V1 ROC-AUC | V2 ROC-AUC | $\Delta$ AUC | V2 Recall | V2 Precision | V2 F1 | Operational Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Groundnut** | LogisticReg | $0.50$ | $0.9080$ | **$0.9224$** | **$+0.0144$** | **$83.3\%$** | $25.0\%$ | $0.3846$ | **Operational (Promising)** |
| **Rice** | RandomForest | $0.45$ | $0.9093$ | **$0.9197$** | **$+0.0104$** | **$88.2\%$** | $46.9\%$ | $0.6038$ | **Operational (Promising)** |
| **Sugarcane** | HistGradBoost | $0.55$ | $0.5744$ | **$0.6365$** | **$+0.0621$** | $50.0\%$ | $37.8\%$ | $0.4308$ | **Experimental** |
| **Cotton(lint)** | RandomForest | $0.50$ | $0.7352$ | **$0.7212$** | $-0.0140$ | $16.7\%$ | $16.7\%$ | $0.1667$ | **Research Prototype** |
| **Maize** | LogisticReg | $0.50$ | $0.8243$ | **$0.8333$** | $+0.0090$ | $0.0\%$ | $0.0\%$ | $0.0000$ | **Experimental** |

---

## 🛡️ Honest Scientific Governance & Known Limitations

In compliance with professional and academic engineering standards, all model constraints are explicitly documented:

1. **Retrospective Evaluation Disclosure:** The 2010–2013 protected test set was previously inspected during exploratory research. Reported generalization scores represent a rigorous **retrospective benchmark** rather than an untouched prospective field trial.
2. **Rainfed Cotton Variance:** Rainfed cotton yield volatility cannot be resolved solely from macro-climate and soil fertility features (Test $R^2 = -0.2394$). Cotton models are strictly designated as **Research Prototypes**.
3. **Maize Hybrid Seed Expansion:** Following the post-2007 hybrid revolution in Tamil Nadu, yields surged statewide. The 2010–2013 test set contains only **2 true positive low-yield cases** out of 113 observations. Consequently, the default $0.50$ decision threshold yields an F1 of $0.0$, despite high ROC-AUC ($0.8333$). Designated as **Experimental**.
4. **Sugarcane Classification Discrimination:** While $0.10\times$ unit scaling restored fallback regression ($R^2 = +0.0511$), classification discrimination remains near chance level ($\text{ROC-AUC} = 0.6365$). Designated as **Experimental**.
5. **Operational Models (Rice & Groundnut):** While demonstrating strong retrospective metrics ($R^2 > 0.19$, $\text{ROC-AUC} > 0.91$), they reflect historical district averages and are not independently validated as on-farm precision advisory systems.

---

## 📁 Repository Structure

```
crop-yield-climate-risk-ml/
├── app.py                                   # Main Streamlit dashboard application
├── inference_v2.py                          # Authoritative V2 Inference Engine
├── test_inference_v2.py                     # V2 Inference test suite (18 automated tests)
├── test_frontend_integration.py             # Frontend and data loader integration tests
├── tn_master_dataset_stage1_ready_v2.csv    # Cleansed V2 master dataset (2,333 records)
├── requirements.txt                         # Python package dependencies
├── .env.example                             # Environment configuration template
├── .gitignore                               # Git ignore rules (prompts/ excluded)
│
├── assets/
│   └── styles.css                           # Analytics SaaS CSS theme (Forest Green)
│
├── ui_pages/                                # Modular dashboard views
│   ├── overview.py                          # Page 1: Overview Dashboard
│   ├── yield_predictor.py                   # Page 2: Scenario Yield Predictor
│   ├── risk_analyzer.py                     # Page 3: Low-Yield Risk Analyzer
│   ├── agro_zones.py                        # Page 4: Agro-Zone Explorer
│   ├── explainability.py                    # Page 5: Explainability & Model Audit
│   └── methodology.py                       # Page 6: Methodology & Documentation
│
├── ui_components/                           # Reusable UI widgets
│   ├── cards.py                             # Stat cards, status pills, epistemic badges
│   ├── charts.py                            # Plotly visualizers (trends, maps, gauges)
│   └── inputs.py                            # Parameter forms with district baseline autofill
│
├── utils/                                   # Data & engine access layer
│   ├── engine_provider.py                   # Cached singleton inference provider
│   └── data_loader.py                       # Cached dataset and metrics loaders
│
├── model_artifacts_v2/                      # Serialized Cleansed ML Models & Metadata
│   ├── model_metadata.json                  # Retrained metrics, schemas, and governance
│   ├── risk_thresholds.json                 # Tau_20 cutoffs and tuned decision thresholds
│   ├── cluster_metadata.json                # K-Means centroids and district coordinates
│   ├── feature_ranges.json                  # Physical bounds and training [p01, p99]
│   ├── cluster_model.joblib                 # StandardScaler + KMeans pipeline
│   ├── yield_model_*.joblib                 # 5 Primary Regression pipelines (Group B)
│   ├── yield_model_fallback_*.joblib        # 5 Fallback Regression pipelines (Group A)
│   ├── risk_model_*.joblib                  # 5 Primary Risk Classifiers (Group B)
│   └── risk_model_fallback_*.joblib         # 5 Fallback Risk Classifiers (Group A)
│
└── ml_audit_results_v2/                     # Audit & Comparison CSV Reports
    ├── stage1_regression_comparison_v1_vs_v2.csv
    ├── stage3_classification_comparison_v1_vs_v2.csv
    └── risk_thresholds_comparison_v1_vs_v2.csv
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python `3.10` or higher
- Git

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/juzerwithaz/crop-yield-climate-risk-ml.git
cd crop-yield-climate-risk-ml

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run the Automated Verification Test Suites
Verify that model pipelines, boundaries, and integrations are completely intact:
```bash
# 1. Run V2 Inference Engine test suite (18 tests)
python test_inference_v2.py

# 2. Run Frontend and Data Loader integration test suite (6 tests)
python test_frontend_integration.py
```

### 4. Launch the Streamlit Analytics Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 💻 Programmatic API Usage

You can use the inference engine directly in Python scripts or research notebooks:

```python
from inference_v2 import CropYieldInferenceEngineV2

# Initialize engine (loads from model_artifacts_v2/)
engine = CropYieldInferenceEngineV2()

# 1. Predict Crop Yield (Group B Primary with 3-year historical lag)
yield_result = engine.predict_yield(
    crop="Rice",
    district="THANJAVUR",
    rainfall=950.0,      # mm/year
    temp=28.2,           # °C
    humidity=71.5,       # %
    ph=7.8,              # pH scale
    n=250.0,             # kg/ha
    p=22.0,              # kg/ha
    k=290.0,             # kg/ha
    yield_lag_avg=3.45   # t/ha (omit or set to None to route to Fallback)
)

print(f"Predicted Yield: {yield_result['predicted_yield']:.2f} {yield_result['unit']}")
print(f"Pipeline Used: {yield_result['feature_group_used']} ({yield_result['model_family']})")
print(f"Operational Status: {yield_result['model_status']}")

# 2. Evaluate Low-Yield Risk (Binary tau_20 cutoff)
risk_result = engine.predict_risk(
    crop="Rice",
    district="THANJAVUR",
    rainfall=650.0,      # Deficit rainfall scenario
    temp=30.5,
    humidity=62.0,
    ph=7.8,
    n=250.0,
    p=22.0,
    k=290.0,
    yield_lag_avg=2.80
)

print(f"Risk Assessment: {risk_result['predicted_risk_class']}")
print(f"Estimated Probability: {risk_result['risk_probability']:.2%} (Uncalibrated)")
print(f"Decision Cutoff (p*): {risk_result['decision_threshold_p']:.2f}")

# 3. Retrieve Agro-Climatic Profile & Cluster
profile = engine.get_district_profile("COIMBATORE")
print(f"Cluster: {profile['cluster_id']} - {profile['cluster_name']}")
print(f"Soil Survey Confidence: {profile['soil_confidence']}")
```

---

## 📜 Epistemic Integrity & Labeling

The platform visually tags every piece of information to ensure total transparency:
- `[Observed Data]`: Historical records and verified official agricultural statistics.
- `[Model Prediction]`: Statistical machine learning outputs from trained algorithms.
- `[Statistical Association]`: Clustering proximity or feature correlation (not causality).
- `[Agronomic Hypothesis]`: External domain explanations requiring field validation.

---

## 📄 License & Acknowledgements

- **License:** Distributed under the [MIT License](LICENSE).
- **Meteorological Data:** Sourced from NASA Prediction Of Worldwide Energy Resources (POWER) project.
- **Agricultural Statistics:** Derived from Tamil Nadu Department of Economics & Statistics and Directorate of Agriculture historical returns.

