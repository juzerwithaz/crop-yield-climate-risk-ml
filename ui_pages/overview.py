"""
overview.py
Page 1: Overview Dashboard for Tamil Nadu Crop Intelligence.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_v2_dataset, get_dataset_summary_stats
from ui_components.cards import render_stat_card, render_callout, render_epistemic_badge
from ui_components.charts import create_statewide_trends_chart, create_soil_confidence_donut

def render_overview_page(engine):
    st.markdown('<div class="badge-epistemic epistemic-observed">[Observed Data & Governance Overview]</div>', unsafe_allow_html=True)
    st.title("🌾 Tamil Nadu Crop Intelligence")
    st.markdown(
        "*Understanding district-level crop yield dynamics, climate patterns, and agricultural risk across Tamil Nadu.*"
    )

    st.markdown("---")

    # High-level summary metrics
    stats = get_dataset_summary_stats()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_stat_card(
            val=f"{stats['districts_count']}",
            label="Administrative Districts",
            subtext="Complete Tamil Nadu spatial coverage",
            epistemic="Observed Data"
        )
    with c2:
        render_stat_card(
            val=f"{stats['crops_count']}",
            label="Principal Crops",
            subtext="Rice, Groundnut, Maize, Cotton, Sugarcane",
            epistemic="Observed Data"
        )
    with c3:
        render_stat_card(
            val=f"{stats['years_range'][0]}–{stats['years_range'][1]}",
            label="Historical Span",
            subtext="17 crop years (Train, Val, Test)",
            epistemic="Observed Data"
        )
    with c4:
        render_stat_card(
            val="4 Zones",
            label="Agro-Climatic Clusters",
            subtext="K-Means (k=4) climate-soil typology",
            epistemic="Statistical Association"
        )

    # Dataset Provenance & Cleansing Counts
    st.markdown("### 📋 Dataset Scope & Quality-Status Audit")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_stat_card(
            val=f"{stats['total_records']:,}",
            label="Total Master Records",
            subtext="Cleansed V2 master dataset rows",
            epistemic="Observed Data"
        )
    with k2:
        render_stat_card(
            val=f"{stats['eligible_records']:,}",
            label="Eligible Primary Records",
            subtext=f"{stats['valid_records']} VALID + {stats['scaled_records']} Scaled 1997",
            epistemic="Observed Data"
        )
    with k3:
        render_stat_card(
            val=f"{stats['total_excluded']}",
            label="Excluded Records",
            subtext=f"{stats['excluded_missing_prod']} missing prod + {stats['excluded_admin_bifurcation']} 2008 bifurcation",
            epistemic="Observed Data"
        )
    with k4:
        render_stat_card(
            val=f"{stats['train_records']} / {stats['validation_records']} / {stats['test_records']}",
            label="Chronological Splits",
            subtext="Train (<=2007) / Val (2008-09) / Test (2010-13)",
            epistemic="Observed Data"
        )

    # Core Findings & V1 vs V2 Breakthroughs
    st.markdown("### 🚀 Key Findings & Cleansing Impacts (V1 Baseline vs V2 Cleansed)")
    f1, f2 = st.columns(2)

    with f1:
        st.markdown("""
        <div class="stat-card">
            <div class="badge-epistemic epistemic-prediction">[Model Prediction]</div>
            <div style="font-weight:700; font-size:1.1rem; color:#1b4332; margin-bottom:0.5rem;">
                Dramatic Recovery in Maize & Sugarcane Fallback
            </div>
            <ul style="font-size:0.9rem; color:#374151; padding-left:1.2rem; margin:0;">
                <li><b>Maize Primary Regression:</b> Test R² surged from <b>-1.50</b> in V1 to <b>+0.0568</b> in V2 (RMSE dropped by <b>1.02 t/ha</b>) after removing the 2008 Coimbatore bifurcation distortion.</li>
                <li><b>Sugarcane Fallback Pipeline:</b> Restored from catastrophic collapse (R² = <b>-24.52</b>, RMSE = 79.19 t/ha) to <b>+0.0511</b> (RMSE = 15.27 t/ha) following 0.10x 1997 unit scaling.</li>
                <li><b>Rice & Groundnut:</b> Both confirmed as <b>operational, promising retrospective models</b> with positive test R² (+0.196 and +0.210) and ROC-AUC > 0.91.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="stat-card">
            <div class="badge-epistemic epistemic-hypothesis">[Agronomic & Governance Caveats]</div>
            <div style="font-weight:700; font-size:1.1rem; color:#92400e; margin-bottom:0.5rem;">
                Documented Scientific Boundaries & Limitations
            </div>
            <ul style="font-size:0.9rem; color:#374151; padding-left:1.2rem; margin:0;">
                <li><b>Retrospective Evaluation Disclosure:</b> The 2010–2013 test set was previously inspected during exploratory research; reported metrics serve as retrospective benchmarks rather than an untouched prospective trial.</li>
                <li><b>Rainfed Cotton Volatility:</b> Cotton regression remains negative (R² = -0.239), showing that macro-climate and soil alone cannot resolve rainfed boll variance.</li>
                <li><b>Maize Hybrid Regime Shift:</b> Tamil Nadu's post-2007 hybrid seed expansion yielded only 2 low-yield cases in the test set, creating extreme imbalance.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Visualizations
    st.markdown("### 📈 Statewide Production Trends & Soil Survey Confidence")
    v_col1, v_col2 = st.columns([1.6, 1])

    df_v2 = load_v2_dataset()

    with v_col1:
        crop_sel = st.selectbox(
            "Filter Crop for Historical Trends",
            ["All Crops"] + sorted(df_v2['Crop'].unique().tolist()),
            key="overview_crop_filter"
        )
        trends_fig = create_statewide_trends_chart(df_v2, selected_crop=crop_sel)
        st.plotly_chart(trends_fig, use_container_width=True)

    with v_col2:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        donut_fig = create_soil_confidence_donut(engine.cluster_metadata)
        st.plotly_chart(donut_fig, use_container_width=True)

    # Research Guidance Callout
    render_callout(
        title="Responsible Agricultural Decision-Making Guidance",
        text=(
            "Predictions generated in this application are statistical machine learning estimates derived from historical "
            "district-level crop reporting (1997–2013) and gridded daily climate aggregations. They do not substitute for "
            "on-farm agronomic advisory, local soil testing, or certified meteorological forecasts."
        ),
        level="info"
    )

