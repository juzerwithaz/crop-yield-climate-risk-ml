# Stage 3 Research Report: Yield-Risk Classification and Explainability

**Date:** 2026-10-29  
**Study Scope:** Tamil Nadu, India (31 Districts, 5 Crops, 1997–2013)  
**Execution Script:** [`12_stage3_yield_risk_classification.py`](file:///c:/Users/Admin/Desktop/Crop_Prod/12_stage3_yield_risk_classification.py)  
**Input Datasets:** [`tn_master_dataset_stage1_ready.csv`](file:///c:/Users/Admin/Desktop/Crop_Prod/tn_master_dataset_stage1_ready.csv) & [`stage2_results/stage2_cluster_assignments.csv`](file:///c:/Users/Admin/Desktop/Crop_Prod/stage2_results/stage2_cluster_assignments.csv)  
**Results Directory:** [`stage3_results/`](file:///c:/Users/Admin/Desktop/Crop_Prod/stage3_results)  
**Git Status:** Local only — zero commits or pushes made (held for user review)  

---

## 1. Executive Summary

1. **Operational Definition of Yield Risk:**
   Binary `High_Yield_Risk` was defined per crop using the **20th percentile of historical yields estimated exclusively from the training period ($\text{Year} \le 2007$)**. Thresholds were strictly fixed across validation ($2008–2009$) and test ($2010–2013$).
   - Cotton (lint): $\tau_{20} = 1.1772\text{ t/ha}$
   - Groundnut: $\tau_{20} = 1.3077\text{ t/ha}$
   - Maize: $\tau_{20} = 1.2341\text{ t/ha}$
   - Rice: $\tau_{20} = 2.7434\text{ t/ha}$
   - Sugarcane: $\tau_{20} = 91.4410\text{ t/ha}$

2. **Final Test Performance (2010–2013):**
   * **Rice (`HistGradientBoosting`, Group B):** Balanced Accuracy $= \mathbf{0.8244}$, Recall $= \mathbf{0.8824}$ (caught 15 of 17 low-yield risk years), Precision $= 0.3750$, $F_1 = 0.5263$, ROC-AUC $= \mathbf{0.9230}$, PR-AUC $= \mathbf{0.7319}$.
   * **Groundnut (`LogisticRegression`, Group B):** Balanced Accuracy $= \mathbf{0.8348}$, Recall $= \mathbf{0.8333}$ (caught 5 of 6 low-yield events), Precision $= 0.2083$, $F_1 = 0.3333$, ROC-AUC $= \mathbf{0.9109}$, PR-AUC $= \mathbf{0.4603}$.
   * **Maize (`LogisticRegression`, Group B):** Balanced Accuracy $= \mathbf{0.7455}$, Recall $= 0.5000$ (1 of 2 caught), Precision $= 0.5000$, $F_1 = 0.5000$, ROC-AUC $= 0.7883$.
   * **Cotton (lint) (`RandomForest`, Group B):** Balanced Accuracy $= \mathbf{0.6433}$, Recall $= 0.3333$ (2 of 6 caught), Precision $= 0.2857$, $F_1 = 0.3077$, ROC-AUC $= 0.7960$.
   * **Sugarcane (`HistGradientBoosting`, Group B):** Balanced Accuracy $= \mathbf{0.5813}$, Recall $= \mathbf{0.6000}$ (15 of 25 caught), Precision $= 0.2632$, $F_1 = 0.3659$, ROC-AUC $= 0.5815$.

3. **Post-Hoc Stage 2 Cluster Risk Findings:**
   Cross-referencing the Stage 2 agro-environmental clusters with empirical yield risk revealed striking, agronomically sound patterns:
   - **Rice Risk:** Highest in **Cluster 0 (Central & Delta Semi-Arid Plains: 31.98%)**; lowest in Cluster 1 (Western Highlands: 4.92%) and Cluster 2 (North-Western Plateau: 3.92%). Despite the Cauvery Delta being the traditional rice belt, tail-end canal water stress and dry-season droughts create frequent low-yield vulnerability.
   - **Cotton Risk:** Heavily concentrated in **Cluster 3 (Coastal Humid Plain: 29.81%)**; lowest in Cluster 2 (Plateau: 2.00%). High coastal relative humidity and heavy late-monsoon rain induce boll rot and vegetative overgrowth.
   - **Sugarcane Risk:** Highest in **Cluster 2 (North-Western Plateau: 33.33%)**; lowest in Cluster 1 (Highlands: 12.28%). Cluster 2 receives the state's lowest rainfall ($850.9\text{ mm}$), severely constraining high-water-demand sugarcane crops.

---

## 2. Research Objective & Problem Definition

While Stage 1 focused on continuous crop yield regression, Stage 3 formulates an **early-warning classification problem**:
Can climate anomalies, soil chemistry, and historical yield momentum identify whether a district will experience a **low-yield event** ($\le 20\text{th percentile}$)?
This operational definition is designed for disaster risk mitigation, crop insurance trigger monitoring, and agricultural extension planning. It serves as an operational proxy for severe production shortfall, not direct farmer economic loss or insurance claims.

---

## 3. Data Quality & Suspected Invalid Zero-Yield Handling

* **Root Cause & Scope:** 35 observations in the dataset have `Yield == 0.0000`, which was traced back to `Production = NaN` in government reporting silently converted to `0.0` by pandas `.sum()`.
* **Exclusion from Primary Analysis:** These 35 rows were excluded as invalid missing-data artifacts, leaving $N = 2,298$ clean observations.
* **Sensitivity Analysis:** When these rows were retained, artificial zeros in 2005 Maize (train) and 2008 Ariyalur/Tiruppur (validation) distorted training loss, but test set rankings (2010–2013, which contained zero invalid zeros) remained stable.

---

## 4. Class Distribution & Lag Availability Quantification

Quantifying observations across chronological splits using $\tau_{20}$:

| Crop | Split | Total Obs ($N$) | High-Risk ($1$) | Normal ($0$) | Prevalence (%) | Observations with Valid Lag | Missing Lag ($K=0$) | Exclusion Loss (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cotton (lint)** | Train | 291 | 59 | 232 | 20.27% | 260 | 31 | 10.65% |
| | Validation | 57 | 9 | 48 | 15.79% | 57 | 0 | 0.00% |
| | Test | 113 | 6 | 107 | 5.31% | 113 | 0 | 0.00% |
| **Groundnut** | Train | 306 | 62 | 244 | 20.26% | 277 | 29 | 9.48% |
| | Validation | 59 | 5 | 54 | 8.47% | 58 | 1 | 1.69% |
| | Test | 122 | 6 | 116 | 4.92% | 122 | 0 | 0.00% |
| **Maize** | Train | 209 | 42 | 167 | 20.10% | 180 | 29 | 13.88% |
| | **Validation** | **52** | **0** | **52** | **0.00%** | **52** | **0** | **0.00%** |
| | Test | 113 | 2 | 111 | 1.77% | 113 | 0 | 0.00% |
| **Rice** | Train | 312 | 63 | 249 | 20.19% | 283 | 29 | 9.29% |
| | Validation | 60 | 13 | 47 | 21.67% | 60 | 0 | 0.00% |
| | Test | 124 | 17 | 107 | 13.71% | 124 | 0 | 0.00% |
| **Sugarcane** | Train | 300 | 60 | 240 | 20.00% | 272 | 28 | 9.33% |
| | Validation | 58 | 9 | 49 | 15.52% | 57 | 1 | 1.72% |
| | Test | 122 | 25 | 97 | 20.49% | 121 | 1 | 0.82% |

*(Saved to `stage3_results/stage3_class_distribution.csv`).*

---

## 5. Methodological Safeguard: Handling Maize Validation

* **The Phenomenon:** In Maize, actual yields across all districts in 2008–2009 were $> 1.35\text{ t/ha}$, which is strictly above the training 20th percentile threshold ($\tau_{20} = 1.2341\text{ t/ha}$). Consequently, there were **zero positive validation cases ($0 / 52$)**.
* **Integrity Guardrail:**
  - Rather than forcing undefined metrics ($F_1 = 0/0$) or using shuffled cross-validation, we performed **training-only chronological validation**:
    - Train-Early: Years $\le 2003$ ($N=108$, 28 positives)
    - Train-Late: Year 2004 ($N=26$, 7 positives)
  - This allowed objective, leakage-safe model comparison within the training era. Logistic Regression with balanced weights emerged as the top performer ($F_1 = 0.9231$ on 2004).
  - Validation metrics for Maize at $\tau_{20}$ are honestly reported as `NaN` / undefined.

---

## 6. Model Selection on Validation Split (2008–2009)

Selected configurations based on validation $F_1$, Balanced Accuracy, and Recall (Group B features, matched lag observations):

| Crop | Selected Model Family | Feature Group | Hyperparameters | Val $F_1$ | Val Balanced Acc | Val Recall | Selection Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cotton (lint)** | **RandomForest** | Group B | `max_depth=5, balanced` | **0.7778** | **0.8681** | **0.7778** | Highest validation $F_1$ and balanced accuracy |
| **Groundnut** | **LogisticRegression** | Group B | $C=1.0\text{, balanced}$ | **0.3636** | **0.7774** | **0.8000** | Superior recall on minority risk class |
| **Maize** | **LogisticRegression** | Group B | $C=1.0\text{, balanced}$ | Undefined* | Undefined* | Undefined* | Selected via chronological train validation ($F_1=0.9231$) |
| **Rice** | **HistGradientBoosting** | Group B | `lr=0.05, max_depth=3` | **0.8148** | **0.8912** | **0.8462** | Outstanding $F_1$ and recall |
| **Sugarcane** | **HistGradientBoosting** | Group B | `lr=0.05, max_depth=3` | **0.4706** | **0.7674** | **0.8889** | Highest recall and balanced accuracy |

*\*Marked undefined due to zero positive targets in the validation split.*  
*(Full comparison of all 45 validation model configurations logged in `stage3_results/stage3_validation_comparison.csv`).*

---

## 7. Final Test Evaluation Metrics (2010–2013 Holdout)

Selected models refitted on combined development data (`train` + `validation`) and evaluated once on `test`:

| Crop | Model Evaluated | Feature Group | Test $N$ | Positives | Balanced Acc | Recall | Precision | $F_1$-Score | ROC-AUC | PR-AUC | DummyPrior $F_1$ (BAcc) | Confusion Matrix (TN, FP, FN, TP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cotton (lint)** | **RandomForest** | Group B | 113 | 6 (5.3%) | **0.6433** | 0.3333 | 0.2857 | **0.3077** | **0.7960** | 0.1937 | 0.0000 (0.5000) | 102, 5, 4, 2 |
| **Groundnut** | **LogisticRegression** | Group B | 122 | 6 (4.9%) | **0.8348** | **0.8333** | 0.2083 | **0.3333** | **0.9109** | 0.4603 | 0.0000 (0.5000) | 97, 19, 1, 5 |
| **Maize** | **LogisticRegression** | Group B | 113 | 2 (1.8%) | **0.7455** | 0.5000 | 0.5000 | **0.5000** | **0.7883** | 0.5204 | 0.0000 (0.5000) | 110, 1, 1, 1 |
| **Rice** | **HistGradientBoosting** | Group B | 124 | 17 (13.7%) | **0.8244** | **0.8824** | 0.3750 | **0.5263** | **0.9230** | **0.7319** | 0.0000 (0.5000) | 82, 25, 2, 15 |
| **Sugarcane** | **HistGradientBoosting** | Group B | 121 | 25 (20.7%) | **0.5813** | **0.6000** | 0.2632 | **0.3659** | **0.5815** | 0.2469 | 0.0000 (0.5000) | 54, 42, 10, 15 |

*(Saved to `stage3_results/stage3_test_metrics.csv` and `stage3_results/stage3_test_predictions.csv`).*

---

## 8. Explainability & Feature Importance Results

1. **Permutation Importance on Validation Set:**
   - **`Yield_Lag_Avg`** was the dominant protective / predictive feature for Rice, Groundnut, and Sugarcane (dropping balanced accuracy by 0.10–0.25 when permuted). A strong multi-year yield track record strongly decreases the likelihood of falling into the lowest quintile.
   - **`Rainfall` & `Humidity`:** High permutation importance for Cotton and Rice. Extreme rainfall deviations directly increased classification risk.
   - **Soil Macronutrients (`N`, `P`, `K`):** Noticeable permutation importance for Groundnut and Sugarcane. Groundnut exhibited strong sensitivity to available phosphorus and potassium levels.
2. **Logistic Regression Coefficients (Groundnut & Maize):**
   - In Groundnut, standardized coefficients showed that `Yield_Lag_Avg` had a large negative log-odds coefficient ($\beta = -2.14$, Odds Ratio $= 0.12$), indicating that higher past yields dramatically reduce the odds of risk.
   - Temperature had a positive log-odds coefficient ($\beta = +0.68$, Odds Ratio $= 1.97$), indicating that elevated thermal stress increases the odds of low-yield events.
   *(Logged in `stage3_results/stage3_feature_importance.csv`).*

---

## 9. Post-Hoc Stage 2 Cluster Risk Synthesis

Cross-tabulating historical low-yield risk across Stage 2 agro-climatic clusters:

| Stage 2 Cluster Name | Distinct Districts | Total Observations | Rice Risk Rate (%) | Cotton Risk Rate (%) | Sugarcane Risk Rate (%) | Groundnut Risk Rate (%) | Maize Risk Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0: Central & Delta Semi-Arid Plains** | 13 | 197 | **31.98%** (63/197) | 8.12% (16/197) | 22.34% (44/197) | 10.15% (20/197) | 6.75% (11/163) |
| **Cluster 1: Western Ghats & Highland** | 4 | 52–61 | 4.92% (3/61) | 16.98% (9/53) | 12.28% (7/57) | **25.00%** (13/52) | 20.00% (8/40) |
| **Cluster 2: North-Western Plateau** | 3 | 45–51 | 3.92% (2/51) | 2.00% (1/50) | **33.33%** (17/51) | 11.76% (6/51) | 15.56% (7/45) |
| **Cluster 3: Coastal Humid Plain** | 11 | 161–187 | 13.37% (25/187) | **29.81%** (48/161) | 14.86% (26/175) | 18.18% (34/187) | 14.29% (18/126) |

*(Saved to `stage3_results/stage3_cluster_risk_summary.csv`).*

---

## 10. Sensitivity Analyses

1. **Threshold Sensitivity ($\tau_{15}$ vs $\tau_{25}$):**
   - When lowering the risk bar to the 15th percentile ($\tau_{15}$), target prevalence drops on test data (Cotton: 4, Groundnut: 6, Rice: 16), but Rice and Groundnut maintained excellent ROC-AUC ($>0.91$).
   - When raising the bar to the 25th percentile ($\tau_{25}$), Sugarcane balanced accuracy rose from $0.5813 \rightarrow 0.6990$ and $F_1$ reached $0.5250$, as the broader target boundary captured clearer separation.
2. **Invalid Zeros Inclusion Sensitivity:**
   - Including the 35 invalid zeros inflated risk prevalence artificially in training (Maize: $+25$ zeros) and validation (Ariyalur/Tiruppur: $+10$ zeros).
   - Because the test set (2010–2013) contained zero invalid zero-yields, test metrics remained identical for Cotton, Groundnut, Rice, and Sugarcane, while Maize test ROC-AUC dropped from $0.7883 \rightarrow 0.6532$ due to distorted decision thresholds.
   *(Logged in `stage3_results/stage3_sensitivity_analysis.csv`).*

---

## 11. Research Limitations & Ethical Transparency

1. **Operational Target Limitation:** The low-yield label ($\le \tau_{20}$) is a statistical quantile proxy. It reflects production shortfall relative to historical district baseline, not absolute farmer economic bankruptcy or flood damage claims.
2. **Prior Inspection of Test Period:** The 2010–2013 period was explored in earlier project code. While model selection strictly adhered to the 2008–2009 validation split, it cannot be characterized as an untouched blind holdout.
3. **Small Sample Constraints in Clusters:** Cluster 1 ($N=4$ districts) and Cluster 2 ($N=3$ districts) have small sample counts. Empirical risk percentages for these clusters must be interpreted as descriptive indicators, not definitive causal laws.

---

## 12. Instructions to Reproduce

Execute the self-contained pipeline from the workspace directory:
```powershell
python 12_stage3_yield_risk_classification.py
```
This script will re-calculate the training thresholds, train all model families, generate test metrics, compute permutation importances, execute the sensitivity analyses, and output all CSVs and diagnostic plots to `stage3_results/`.

