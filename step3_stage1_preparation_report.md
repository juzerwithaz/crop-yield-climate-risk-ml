# Step 3 Preparation Report: Leakage-Safe Stage 1 Dataset

**Date:** 2026-10-29  
**Script Created:** `09_prepare_stage1_dataset.py`  
**Dataset Generated:** `tn_master_dataset_stage1_ready.csv`  
**Verified Source File:** `tn_master_dataset_weather_corrected.csv`  
**Git Status:** Untracked locally — zero commits or pushes made (held for user review)  

---

## 1. Input Validation Results

Before preparing the dataset, `tn_master_dataset_weather_corrected.csv` was re-verified from disk:
* **Row Count:** 2,333 observations.
* **Columns (19):** `District`, `Year`, `Crop`, `Area`, `Production`, `Yield`, `Rainfall`, `Temp`, `Humidity`, `pH`, `N`, `P`, `K`, `Soil_Confidence`, `Valid_Months`, `Complete_Year`, `Rainfall_Original`, `Temp_Original`, `Humidity_Original`.
* **Key Uniqueness:** Zero duplicate `(District, Year, Crop)` keys.
* **Missing Values:** Zero nulls across all 19 columns.
* **Target:** `Yield` is numeric (`float64`) and complete.
* **Weather & Soil Features:** All numeric, with physical ranges validated in Steps 1 & 2.
* **Categorical Integrity:** `Soil_Confidence` intact (`zone-avg`: 1,489, `real`: 751, `low-confidence`: 93); all rows have `Valid_Months == 12` and `Complete_Year == True`.

---

## 2. Historical Feature Definition: `Yield_Lag_Avg`

### Mathematical Definition
For an observation belonging to district $d$, crop $c$, and calendar year $t$:
$$\mathcal{H}_{d,c,t} = \{ \text{Yield}_{d,c,y} \mid y \in \{t-1, t-2, t-3\} \text{ and } (d,c,y) \in \mathcal{D} \}$$

$$\text{Yield\_Lag\_Years\_Available}_{d,c,t} = |\mathcal{H}_{d,c,t}| = K \in \{0, 1, 2, 3\}$$

$$\text{Yield\_Lag\_Avg}_{d,c,t} = \begin{cases} 
\frac{1}{K} \sum_{y \in \mathcal{H}} \text{Yield}_{d,c,y} & \text{if } K > 0 \\ 
\text{NaN} & \text{if } K = 0 
\end{cases}$$

### Strict Anti-Leakage Guardrails
1. **Zero Target Leakage:** Current year $t$ is strictly excluded from its own lag feature.
2. **Strictly Backward-Looking:** Future years ($t+1, t+2, \dots$) are never accessed.
3. **Partition Isolation:** Yields are grouped strictly by `(District, Crop)`. No cross-district or cross-crop data bleeding.
4. **NaN Preservation:** When $K=0$ (e.g. the first recorded year of a series), `Yield_Lag_Avg` is explicitly assigned `NaN`. It is **not** imputed with zero, global averages, or target values.
5. **No Row Dropping:** All 2,333 observations are preserved in the ready dataset.

---

## 3. Handling of Missing Calendar Years

In our continuity scan, **32 of the 154 district-crop series contain missing intervening calendar years**.
* Traditional row-based sliding windows (e.g., `shift(1).rolling(3)`) incorrectly pull in older years (such as $t-4$ or $t-5$) as if they were consecutive.
* Our calendar-indexed matching queries the exact years $\{t-1, t-2, t-3\}$. If year $t-1$ is absent from the historical records, only the available years within that exact 3-year window are averaged, and `Yield_Lag_Years_Available` accurately records the sample count $K$.

---

## 4. Chronological Validation Split Metadata

Fixed calendar split assigned to each observation:
* **`train`** ($\text{Year} \le 2007$): **1,443 rows** (61.85%)
* **`validation`** ($2008 \le \text{Year} \le 2009$): **296 rows** (12.69%)
* **`test`** ($2010 \le \text{Year} \le 2013$): **594 rows** (25.46%)

**Total:** 2,333 rows. Zero unassigned or overlapping splits.

---

## 5. Missing-Value and Availability Distribution

| `Yield_Lag_Years_Available` | Row Count | Percentage | `Yield_Lag_Avg` Status |
| :---: | :---: | :---: | :---: |
| **0** | **159** | 6.82% | `NaN` (Strictly preserved, no history) |
| **1** | **169** | 7.24% | Valid float (1 prior year) |
| **2** | **223** | 9.56% | Valid float (2 prior years) |
| **3** | **1,782** | 76.38% | Valid float (Full 3-year history) |
| **Total** | **2,333** | **100.0%** | **159 NaNs, 2,174 valid floats** |

---

## 6. Manual Verification Cases

### Case 1: Early Observations (Insufficient History) — Thanjavur Rice
| District | Crop | Year | Actual Yield | `Yield_Lag_Avg` | `Years_Available` | Split | Calculation Details |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| THANJAVUR | Rice | 1997 | 2.6985 | `NaN` | 0 | train | No prior years exist |
| THANJAVUR | Rice | 1998 | 3.4653 | 2.6985 | 1 | train | $= \text{Yield}_{1997}$ |
| THANJAVUR | Rice | 1999 | 3.2291 | 3.0819 | 2 | train | $= (2.6985 + 3.4653)/2$ |
| THANJAVUR | Rice | 2000 | 3.4449 | 3.1310 | 3 | train | $= (2.6985 + 3.4653 + 3.2291)/3$ |

### Case 2: Full 3-Year History — Madurai Groundnut
| District | Crop | Year | Actual Yield | `Yield_Lag_Avg` | `Years_Available` | Split | Calculation Details |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| MADURAI | Groundnut | 2005 | 1.5992 | 1.4249 | 3 | train | Mean of 2002, 2003, 2004 |
| MADURAI | Groundnut | 2006 | 2.1323 | 1.4679 | 3 | train | Mean of 2003, 2004, 2005 |
| MADURAI | Groundnut | 2007 | 1.8709 | 1.6884 | 3 | train | Mean of 2004, 2005, 2006 |
| MADURAI | Groundnut | 2008 | 1.9814 | 1.8675 | 3 | validation | Mean of 2005, 2006, 2007 |

### Case 3: Missing Intervening Calendar Year (Year 2001 Missing) — Coimbatore Maize
| District | Crop | Year | Actual Yield | `Yield_Lag_Avg` | `Years_Available` | Split | Calculation Details |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| COIMBATORE | Maize | 1999 | 1.1253 | 1.1241 | 2 | train | Uses 1997, 1998 |
| COIMBATORE | Maize | 2000 | 1.1254 | 1.1245 | 3 | train | Uses 1997, 1998, 1999 |
| COIMBATORE | Maize | **2002** | 1.1249 | **1.1254** | **2** | train | **2001 missing! Uses exact {1999, 2000}** |
| COIMBATORE | Maize | **2003** | 1.1251 | **1.1251** | **2** | train | **2001 missing! Uses exact {2000, 2002}** |
| COIMBATORE | Maize | **2004** | 1.1250 | **1.1250** | **2** | train | **2001 missing! Uses exact {2002, 2003}** |
| COIMBATORE | Maize | 2005 | 0.0000 | 1.1250 | 3 | train | Uses exact {2002, 2003, 2004} |

---

## 7. Input vs. Reloaded Output Preservation Checks

After saving to disk, `tn_master_dataset_stage1_ready.csv` was reloaded and compared directly against `tn_master_dataset_weather_corrected.csv`:
* **Row Count:** 2,333 $\rightarrow$ 2,333 (**Exact Match**).
* **Column Count:** 19 original columns + 3 new columns = **22 columns total**.
* **Source Columns Check:**
  All 19 original columns (`District`, `Year`, `Crop`, `Area`, `Production`, `Yield`, `Rainfall`, `Temp`, `Humidity`, `pH`, `N`, `P`, `K`, `Soil_Confidence`, `Valid_Months`, `Complete_Year`, `Rainfall_Original`, `Temp_Original`, `Humidity_Original`) matched bit-for-bit (0 mismatches).
* **No File Overwrites:** No pre-existing files in the repository were overwritten or modified.

---

## 8. Known Limitations

1. **Prior Inspection of Test Set:** The 2010–2013 test set was examined during preliminary exploratory data analysis and legacy script execution (`02_stage1_final_model..py`). While chronological split boundaries are strictly respected, it cannot be considered completely unseen data in the strictest double-blind experimental sense.
2. **Missing History in Early Years:** 159 rows have `Yield_Lag_Years_Available == 0` (mostly year 1997). When training tree models or linear models in Stage 1, these observations will need an explicit masking or modeling decision (e.g. separate baseline model or filtering) rather than naive zero-imputation.

---

## 9. Confirmation Statement

* **NO machine learning models were trained, evaluated, or fitted.**
* **NO hyperparameters were tuned.**
* **Stage 2 (clustering) and Stage 3 (explainability) have not been initiated.**
* **ZERO Git commits or pushes were executed (per user instruction).**

