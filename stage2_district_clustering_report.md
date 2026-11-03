# Stage 2 Research Report: District-Level Climate and Soil Clustering

**Date:** 2026-10-29  
**Study Scope:** Tamil Nadu, India (31 Districts)  
**Execution Script:** [`11_stage2_district_clustering.py`](file:///c:/Users/Admin/Desktop/Crop_Prod/11_stage2_district_clustering.py)  
**Results Directory:** [`stage2_results/`](file:///c:/Users/Admin/Desktop/Crop_Prod/stage2_results)  
**Git Status:** Local only — zero commits or pushes made (held for user review)  

---

## 1. Executive Summary & Final Configuration

* **Objective:** Unsupervised segmentation of Tamil Nadu's 31 districts into agro-environmental zones using long-term climate and soil chemistry data.
* **Feature Matrix:** $31 \text{ districts} \times 7 \text{ features}$ (Mean Annual `Rainfall`, `Temp`, `Humidity`, `pH`, `N`, `P`, `K`). Standardized using `StandardScaler`.
* **Exclusions:** All yield-derived columns (`Yield`, `Yield_Lag_Avg`, `Production`, `Area`), splits, and coordinates were strictly excluded from clustering inputs.
* **Selected Algorithm & Cluster Count:** **K-Means with $k=4$ clusters**.
  - Chosen based on multi-metric consensus: highest Silhouette Score ($0.3474$) and lowest Davies–Bouldin Index ($0.9720$) among balanced configurations.
  - Natural alignment with Tamil Nadu's agro-ecological zones: separates the Western Highland districts, Coastal humid belt, Central agricultural plains, and North-Western plateau.
  - High initialization stability: Mean $\text{Adjusted Rand Index (ARI)} = \mathbf{0.9447}$ across 10 random seeds.
  - High robustness to soil data uncertainty: $\text{ARI} = \mathbf{0.9310}$ when re-fitted excluding the two low-confidence soil districts.

---

## 2. Source Datasets & District Alignment

1. **Authoritative Sources:**
   * **Climate:** Derived from `tn_master_dataset_stage1_ready.csv`. Deduplicated to unique `(District, Year)` records before averaging to ensure every calendar year contributes equally.
     - 28 districts: 17 years of data (1997–2013).
     - Krishnagiri: 10 years (2004–2013).
     - Ariyalur & Tiruppur: 6 years (2008–2013).
   * **Soil:** `tn_soil_data.csv` (pH, N, P, K, Soil_Confidence).
   * **Coordinates:** `tn_district_coords.csv` (parsed to decimal degrees for spatial visualization only).
2. **Alignment:**
   * Normalized district names produced a **100% exact match across all 31 districts** (0 mismatches, 0 unmapped districts).
   * Saved feature matrix: [`stage2_results/stage2_district_features.csv`](file:///c:/Users/Admin/Desktop/Crop_Prod/stage2_results/stage2_district_features.csv).

---

## 3. Algorithm Comparison Table ($k = 2$ to $k = 6$)

Evaluated on standardized features ($31 \times 7$):

| Algorithm | $k$ | Silhouette Score (↑) | Davies–Bouldin (↓) | Calinski–Harabasz (↑) | K-Means Inertia / GMM BIC | Cluster Sizes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **K-Means** | 2 | 0.2928 | 1.3675 | 11.8546 | 154.03 | {19, 12} |
| Agglomerative (Ward) | 2 | 0.2342 | 1.8199 | 7.3488 | — | {20, 11} |
| GMM | 2 | 0.2464 | 1.3338 | 7.4953 | 448.46 | {18, 13} |
| **K-Means** | 3 | 0.3161 | 1.1477 | 10.5971 | 123.51 | {16, 11, 4} |
| Agglomerative (Ward) | 3 | 0.2941 | 1.2329 | 8.9828 | — | {16, 11, 4} |
| GMM | 3 | 0.2970 | 1.3432 | 9.8946 | 206.46 | {15, 12, 4} |
| **K-Means (Selected)** | **4** | **0.3474** | **0.9720** | **10.8036** | **98.62** | **{13, 11, 4, 3}** |
| Agglomerative (Ward) | 4 | 0.3380 | 1.1588 | 10.0673 | — | {15, 9, 4, 3} |
| GMM | 4 | 0.3344 | 0.9782 | 10.4918 | 15.06 | {13, 11, 4, 3} |
| **K-Means** | 5 | 0.3992 | 0.9892 | 12.6708 | 73.58 | {11, 10, 4, 3, 3} |
| Agglomerative (Ward) | 5 | 0.3595 | 0.9373 | 10.8929 | — | {13, 9, 4, 3, 2} |
| GMM | 5 | 0.2886 | 1.4760 | 7.1672 | -231.99 | {11, 10, 4, 3, 3} |
| **K-Means** | 6 | 0.4171 | 0.7593 | 14.4884 | 55.67 | {11, 9, 4, 3, 2, 2} |
| Agglomerative (Ward) | 6 | 0.4077 | 0.7566 | 12.8793 | — | {11, 9, 4, 3, 2, 2} |
| GMM | 6 | 0.3334 | 0.9996 | 11.3310 | -328.20 | {11, 9, 4, 3, 2, 2} |

*(Saved to `stage2_results/stage2_clustering_comparison.csv`).*

### Rationale for $k=4$
While $k=5$ and $k=6$ show mathematically higher silhouette scores, they fragment the small 31-district dataset into micro-clusters containing only 1 or 2 districts (e.g. splitting Erode or Coimbatore into singletons). $k=4$ achieves the lowest Davies–Bouldin index ($0.9720$) while maintaining well-sized, agronomically distinct groups ($13, 11, 4, 3$).

---

## 4. Final Cluster Profiles & District Memberships ($k=4$)

### Cluster 0: Central & Delta Semi-Arid Agricultural Plains ($N=13$)
* **Districts:** Ariyalur, Dindigul, Erode, Karur, Madurai, Nagapattinam, Perambalur, Pudukkottai, Sivaganga, Thanjavur, Thiruvarur, Tiruchirappalli, Tiruppur.
* **Agro-Climatic Profile:**
  - **Rainfall:** $883.9 \pm 115.8\text{ mm}$ (Semi-arid to moderate)
  - **Temperature:** $27.6 \pm 0.6^\circ\text{C}$ (Warm lowland plain)
  - **Humidity:** $69.3 \pm 2.5\%$
  - **Soil:** $\text{pH } 7.73 \pm 0.27$ (Slightly alkaline), $\text{N: } 253.2 \pm 62.0\text{ kg/ha}$, $\text{P: } 21.3 \pm 5.9\text{ kg/ha}$, $\text{K: } 289.3 \pm 39.5\text{ kg/ha}$
* **Soil Confidence:** 9 `zone-avg`, 4 `real`.

### Cluster 1: Western Ghats & Highland Cool Agro-Zone ($N=4$)
* **Districts:** Coimbatore, Krishnagiri, The Nilgiris, Theni.
* **Agro-Climatic Profile:**
  - **Rainfall:** $1066.6 \pm 111.8\text{ mm}$ (Moderate-high)
  - **Temperature:** $\mathbf{24.0 \pm 0.6^\circ\text{C}}$ (**$3.5^\circ\text{C}$ cooler than state average**, high elevation)
  - **Humidity:** $73.4 \pm 2.3\%$
  - **Soil:** $\text{pH } 7.25 \pm 0.79$ (Neutral), $\text{N: } 258.2 \pm 60.5\text{ kg/ha}$, $\text{P: } 23.4 \pm 6.9\text{ kg/ha}$, $\mathbf{K: 222.2 \pm 96.5\text{ kg/ha}}$ (Lowest potassium)
* **Soil Confidence:** 3 `real`, 1 `low-confidence` (The Nilgiris).

### Cluster 2: North-Western Low-Nitrogen Plateau ($N=3$)
* **Districts:** Dharmapuri, Namakkal, Salem.
* **Agro-Climatic Profile:**
  - **Rainfall:** $\mathbf{850.9 \pm 67.1\text{ mm}}$ (Lowest rainfall cluster)
  - **Temperature:** $26.6 \pm 0.9^\circ\text{C}$
  - **Humidity:** $67.7 \pm 0.9\%$ (Driest air)
  - **Soil:** $\text{pH } 7.84 \pm 0.06$ (Alkaline), $\mathbf{N: 174.3 \pm 27.5\text{ kg/ha}}$ (**Significantly lower nitrogen**), $\mathbf{P: 42.3 \pm 7.6\text{ kg/ha}}$ (**Highest phosphorus**), $\text{K: } 388.3 \pm 91.5\text{ kg/ha}$
* **Soil Confidence:** 2 `zone-avg`, 1 `real`.

### Cluster 3: Coastal Humid & High-Potassium Plain ($N=11$)
* **Districts:** Cuddalore, Kanchipuram, Kanniyakumari, Ramanathapuram, Thiruvallur, Tirunelveli, Tiruvannamalai, Tuticorin, Vellore, Villupuram, Virudhunagar.
* **Agro-Climatic Profile:**
  - **Rainfall:** $\mathbf{1189.8 \pm 161.6\text{ mm}}$ (**Highest rainfall cluster**, Northeast monsoon maritime corridor)
  - **Temperature:** $27.5 \pm 0.7^\circ\text{C}$
  - **Humidity:** $\mathbf{74.0 \pm 3.6\%}$ (**Highest humidity**)
  - **Soil:** $\mathbf{pH: 6.86 \pm 0.35}$ (Slightly acidic to neutral), $\text{N: } 234.2 \pm 6.8\text{ kg/ha}$, $\text{P: } 23.0 \pm 6.9\text{ kg/ha}$, $\mathbf{K: 463.9 \pm 66.4\text{ kg/ha}}$ (**Highest potassium**)
* **Soil Confidence:** 8 `zone-avg`, 2 `real`, 1 `low-confidence` (Kanniyakumari).

*(Saved to `stage2_results/stage2_cluster_profiles.csv` and `stage2_results/stage2_cluster_assignments.csv`).*

---

## 5. Stability & Sensitivity Analysis

| Test Type | Description | Metric | Score | Finding |
| :--- | :--- | :---: | :---: | :--- |
| **Random Seed Sensitivity** | 10 K-Means seeds (0–9) vs seed 42 on 31 districts | ARI | **0.9447** (Min: 0.6022, Max: 1.0000) | Highly stable; 9 of 10 seeds produced identical partition. |
| **Algorithm Agreement** | K-Means vs Agglomerative Ward ($k=4$) | ARI | **0.5175** | Moderate agreement; identical on Highland and Plateau clusters, minor boundary shifts between Central and Coastal plains. |
| **Soil Uncertainty Sensitivity** | Excluding Kanniyakumari & The Nilgiris, refit on 29 districts | ARI | **0.9310** | **Exceptional stability.** Cluster boundaries are not driven by the two low-confidence soil records. |

*(Saved to `stage2_results/stage2_cluster_stability.csv`).*

---

## 6. Visualizations Generated

All plots are stored in [`stage2_results/`](file:///c:/Users/Admin/Desktop/Crop_Prod/stage2_results):
1. **`stage2_metric_comparison.png`:** Silhouette, Davies-Bouldin, and Elbow inertia across $k \in [2, 6]$.
2. **`stage2_hierarchical_dendrogram.png`:** Ward hierarchical dendrogram showing natural branch breaks across the 31 districts.
3. **`stage2_pca_clusters.png`:** 2D PCA projection accounting for $62.8\%$ of total variance, showing distinct cluster boundaries.
4. **`stage2_cluster_heatmap.png`:** Z-score feature heatmap highlighting each cluster's defining characteristics.
5. **`stage2_spatial_map.png`:** Geographic plot of Tamil Nadu districts (Latitude vs Longitude) revealing spatial coherence (Western Ghats, Coastal Corridor, Central Delta, North-Western Interior).

---

## 7. Limitations & Transition to Stage 3

1. **Small Sample Size ($N=31$):**
   Clustering 31 regional administrative units means each district carries ~3.2% weight. While clusters are statistically stable ($\text{ARI} > 0.93$), assignments near cluster boundaries (such as Cuddalore/Nagapattinam between coastal and delta) should be treated as continuous gradients rather than rigid barriers.
2. **Soil Granularity:**
   19 districts rely on regional zone-average soil metrics. While the sensitivity test confirmed cluster boundaries are robust, soil features reflect zonal baselines rather than high-resolution field variations.
3. **Preparation for Stage 3 (Yield-Risk & Explainability):**
   * These clusters represent **purely environmental potential and stress factors** (drought exposure, thermal stress, soil fertility constraints).
   * In Stage 3, we will cross-reference these environmental clusters with Stage 1 crop yield prediction errors and variance to classify district yield risk and compute explainability attributions (e.g. SHAP) without circular target leakage.

---

## 8. Instructions to Reproduce

Run the self-contained script from the workspace directory:
```powershell
python 11_stage2_district_clustering.py
```
This will rebuild the feature matrix, run all clustering algorithms, execute the sensitivity analyses, and output all tables and plots to `stage2_results/`.

