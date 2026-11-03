"""
methodology.py
Page 6: About & Methodology for Tamil Nadu Crop Intelligence.
"""

import streamlit as st
from ui_components.cards import render_callout, render_epistemic_badge

def render_methodology_page():
    st.markdown('<div class="badge-epistemic epistemic-observed">[Research Methodology & Reproducibility Guide]</div>', unsafe_allow_html=True)
    st.title("📚 Research Methodology, Provenance & Governance")
    st.markdown(
        "Complete technical documentation of data lineage, cleansing specifications, temporal validation design, "
        "and reproducible experimental protocols."
    )

    st.markdown("---")

    # 1. Geographic & Agricultural Scope
    st.markdown("### 1. Spatial & Agricultural Domain")
    st.markdown("""
    - **Geographic Scope:** The state of Tamil Nadu, India, spanning **31 administrative districts** (based on the administrative boundaries active during the 1997–2013 observation window).
    - **Crop Coverage:** Five principal agricultural crops representing distinct agronomic regimes:
      1. **Rice (Paddy):** Highly irrigated, Cauvery delta staple.
      2. **Groundnut (Peanut):** Semi-arid oilseed cultivated under mixed irrigated/rainfed conditions.
      3. **Maize (Corn):** Rapidly expanding cereal subject to a dramatic post-2007 hybrid regime shift.
      4. **Cotton(lint):** Volatile cash crop predominantly grown under rainfed conditions.
      5. **Sugarcane:** Long-duration (10–14 month) perennial commercial crop requiring high water inputs.
    - **Temporal Window:** 17 crop years from **1997 to 2013**, totaling **2,333 master observation records**.
    """)

    # 2. Data Cleansing & Provenance Audit
    st.markdown("### 2. Data Provenance & Cleansing Protocols (V2 Specification)")
    st.markdown("""
    The transition from the V1 exploratory dataset to the authoritative **V2 Cleansed Dataset** (`tn_master_dataset_stage1_ready_v2.csv`) resolved four critical data anomalies:
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <div style="font-weight:700; color:#1b4332;">A. 1997 Sugarcane Production Unit Scaling</div>
            <p style="font-size:0.875rem; color:#374151; margin-top:0.35rem;">
                Historical 1997 Sugarcane records exhibited unphysical yields (~1,000 t/ha). 
                Multi-source auditing verified a 10x reporting unit mismatch. Production was scaled by <b>0.10x</b> 
                (area-weighted yield adjusted from 1,085.5 to <b>108.55 t/ha</b>). 
                Original values were preserved in <code>Production_Source</code> and <code>Yield_Source</code> with documented provenance uncertainty.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="stat-card">
            <div style="font-weight:700; color:#1b4332;">B. Coimbatore 2008 Administrative Bifurcation</div>
            <p style="font-size:0.875rem; color:#374151; margin-top:0.35rem;">
                In 2008–2009, Tiruppur district was bifurcated from Coimbatore and Erode. 
                In 2008, Coimbatore reported aggregate production across undivided boundaries while land area was bifurcated, 
                generating artificial yield spikes (e.g. 25.68 t/ha in Maize). 
                All 5 Coimbatore 2008 crop records are flagged as <code>EXCLUDED_ADMIN_BIFURCATION</code> and omitted from model training.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="stat-card">
            <div style="font-weight:700; color:#1b4332;">C. Correction of 35 Missing-Production Zero Yields</div>
            <p style="font-size:0.875rem; color:#374151; margin-top:0.35rem;">
                Lineage tracing back to raw source tables revealed that 35 historical district-crop records 
                with <code>Yield = 0.0</code> actually possessed <code>NaN</code> in raw production reports. 
                Previous aggregation scripts converted missing values to zero. 
                These records are flagged as <code>EXCLUDED_SOURCE_MISSING_PRODUCTION</code> and active yields restored to missing.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="stat-card">
            <div style="font-weight:700; color:#1b4332;">D. Non-Leaking Calendar Lag Features</div>
            <p style="font-size:0.875rem; color:#374151; margin-top:0.35rem;">
                Historical yield averages (<code>Yield_Lag_Avg</code>) were regenerated strictly using previous calendar crop years 
                (t-1, t-2, t-3). Rolling computations exclude any record with administrative exclusions or artificial zeros, 
                preventing data leakage and artifact propagation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 3. NASA POWER Rainfall Clarification (Safeguard 2)
    st.markdown("### 3. Climate Feature Engineering & NASA POWER Processing")
    render_callout(
        title="Clarification on Meteorological Data Processing (Safeguard 2)",
        text=(
            "Gridded meteorological features were sourced from NASA POWER daily satellite-derived reanalysis. "
            "The rainfall pipeline performs <b>daily precipitation aggregation and unit conversion/alignment</b> "
            "(summing daily mm to annual/seasonal cumulative precipitation). It is <b>not</b> a ground-station bias correction. "
            "Temperature and relative humidity are derived as annual arithmetic means across complete calendar cycles."
        ),
        level="info"
    )

    # 4. Temporal Validation & Disclosure
    st.markdown("### 4. Chronological Validation & Retrospective Disclosure")
    st.markdown("""
    To avoid temporal data leakage common in spatial k-fold cross-validation, models are evaluated on a strict chronological split:
    - **Training Split (1997–2007, $N = 1,443$):** Used for pipeline fitting and calculation of the 20th percentile yield-risk cutoff ($\tau_{20}$).
    - **Validation Split (2008–2009, $N = 296$):** Used exclusively for model selection and probability decision threshold tuning ($p^*$).
    - **Protected Test Split (2010–2013, $N = 594$):** Used for final retrospective performance evaluation.
    """)

    render_callout(
        title="Scientific Disclosure: Prior Test Set Inspection",
        text=(
            "The 2010–2013 evaluation period was previously inspected during exploratory research prior to data cleansing. "
            "In accordance with rigorous ML auditing standards, all reported test metrics must be treated as a "
            "<b>retrospective benchmark</b> rather than an untouched prospective trial."
        ),
        level="warning"
    )

    # 5. Reproducibility & Test Suite Instructions
    st.markdown("### 5. Reproducibility & Automated Test Suite")
    st.markdown("""
    The entire machine learning and inference pipeline is deterministically reproducible from the project root:
    ```bash
    # 1. Execute the V2 Inference Test Suite (18 tests verifying boundaries, fallbacks & governance)
    python test_inference_v2.py

    # 2. Verify backward-compatibility against original V1 test suite (13 tests)
    python test_inference.py

    # 3. Launch the Streamlit Frontend
    streamlit run app.py
    ```
    """)

