"""
explainability.py
Page 5: Explainability & Model Performance for Tamil Nadu Crop Intelligence.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import (
    load_regression_comparison,
    load_classification_comparison,
    extract_v2_model_weights_summary,
    load_v1_feature_importance_benchmark
)
from ui_components.cards import render_stat_card, render_callout, render_epistemic_badge
from ui_components.charts import create_regression_comparison_bar, create_feature_importance_bar

def render_explainability_page(engine):
    st.markdown('<div class="badge-epistemic epistemic-prediction">[Model Performance & Transparency Audit]</div>', unsafe_allow_html=True)
    st.title("🔬 Explainability & Model Performance Audit")
    st.markdown(
        "Comprehensive, honest scientific comparison of model generalization before and after data cleansing (V1 Baseline vs V2 Cleansed), "
        "including genuine V2 feature importances and documented performance regressions."
    )

    st.markdown("---")

    # Tabbed layout for clear organization
    tab_reg, tab_cls, tab_imp, tab_status = st.tabs([
        "1. Regression Audit (Stage 1)",
        "2. Risk Classification Audit (Stage 3)",
        "3. Feature Importance & Weights",
        "4. Operational Governance Status"
    ])

    with tab_reg:
        st.subheader("Stage 1: Crop Yield Regression Evaluation (Protected Test: 2010–2013)")
        st.markdown(
            "Comparison of primary Group B (with lag) and fallback Group A (without lag) models across all 5 crops. "
            "Evaluated strictly on the protected test split with cleansed records excluded."
        )

        reg_df = load_regression_comparison()

        # Highlight chart
        fig_reg = create_regression_comparison_bar(reg_df)
        st.plotly_chart(fig_reg, use_container_width=True)

        # Detailed metrics table
        display_reg = reg_df[[
            'Crop', 'Model_Type', 'Model_V1', 'Model_V2',
            'Test_R2_V1', 'Test_R2_V2', 'Delta_R2',
            'Test_RMSE_V1', 'Test_RMSE_V2', 'Delta_RMSE',
            'Test_MAE_V1', 'Test_MAE_V2', 'Model_Status_V2'
        ]].copy()

        # Format numbers for clean presentation
        for col in ['Test_R2_V1', 'Test_R2_V2', 'Delta_R2']:
            display_reg[col] = display_reg[col].map(lambda x: f"{x:+.4f}" if pd.notnull(x) else "")
        for col in ['Test_RMSE_V1', 'Test_RMSE_V2', 'Delta_RMSE', 'Test_MAE_V1', 'Test_MAE_V2']:
            display_reg[col] = display_reg[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

        st.dataframe(display_reg, use_container_width=True)

        # Regressions & Caveats callout
        render_callout(
            title="Honest Audit Accounting: Where Models Regressed or Remained Negative",
            text=(
                "• <b>Cotton(lint) Regression:</b> Test R² deteriorated from -0.1802 in V1 to <b>-0.2394</b> in V2. "
                "Macro-environmental features alone cannot explain rainfed cotton yield variance; correctly held as a Research Prototype.<br/>"
                "• <b>Maize Breakthrough:</b> Test R² surged by <b>+1.5560</b> (from -1.499 to +0.0568) after eliminating the 2008 Coimbatore bifurcation distortion.<br/>"
                "• <b>Sugarcane Fallback Recovery:</b> Restored from catastrophic collapse (R² = -24.52) to <b>+0.0511</b> following 0.10x unit correction."
            ),
            level="warning"
        )

    with tab_cls:
        st.subheader("Stage 3: Binary Yield-Risk Classification Audit (Protected Test: 2010–2013)")
        st.markdown(
            "Evaluation of low-yield risk prediction against the 20th percentile training yield threshold (τ₂₀)."
        )

        cls_df = load_classification_comparison()

        display_cls = cls_df[[
            'Crop', 'Model_Type', 'Model_V2', 'Prob_Threshold_V2',
            'ROC_AUC_V1', 'ROC_AUC_V2', 'Delta_ROC_AUC',
            'Balanced_Accuracy_V1', 'Balanced_Accuracy_V2', 'Delta_Balanced_Acc',
            'F1_V1', 'F1_V2', 'Brier_Score_V2', 'Model_Status_V2'
        ]].copy()

        for col in ['ROC_AUC_V1', 'ROC_AUC_V2', 'Delta_ROC_AUC', 'Balanced_Accuracy_V1', 'Balanced_Accuracy_V2', 'Delta_Balanced_Acc', 'F1_V1', 'F1_V2', 'Brier_Score_V2']:
            display_cls[col] = display_cls[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")

        st.dataframe(display_cls, use_container_width=True)

        render_callout(
            title="Classification Vulnerabilities & Imbalance Warnings",
            text=(
                "• <b>Maize Test Imbalance:</b> The 2010–2013 test set contains only <b>2 positive low-yield cases</b> due to the post-2007 hybrid expansion. "
                "Consequently, the default 0.50 probability cutoff yields an F1 of 0.00, despite high ROC-AUC (0.8333). Flagged as Experimental.<br/>"
                "• <b>Sugarcane Discrimination:</b> Test ROC-AUC improved (+0.0621) but remains at <b>0.6365</b>, close to chance level. Flagged as Experimental."
            ),
            level="warning"
        )

    with tab_imp:
        st.subheader("Model Feature Importance & Weight Analysis")
        st.markdown(
            "Inspection of feature importance values extracted directly from the genuine serialized V2 model artifacts "
            "(`model_artifacts_v2/`), alongside the historical Stage 3 baseline benchmark."
        )

        v2_weights = extract_v2_model_weights_summary()

        imp_crop = st.selectbox(
            "Select Crop to Inspect Genuine V2 Feature Weights:",
            ['Rice', 'Groundnut', 'Maize', 'Cotton(lint)', 'Sugarcane'],
            key="imp_crop_select"
        )

        c_imp1, c_imp2 = st.columns(2)

        with c_imp1:
            key_reg = f"{imp_crop}_regression"
            if key_reg in v2_weights:
                df_r = v2_weights[key_reg]
                m_family = df_r['Model_Family'].iloc[0]
                m_method = df_r['Method'].iloc[0]
                fig_imp_reg = create_feature_importance_bar(
                    df_r,
                    title=f"{imp_crop} V2 Regression Weights ({m_family})"
                )
                st.plotly_chart(fig_imp_reg, use_container_width=True)
                st.caption(f"Method: {m_method} | Artifact: <code>yield_model_{imp_crop.lower()[:4]}.joblib</code>")
            else:
                st.info(f"V2 feature weights not applicable or unavailable for {imp_crop} regression.")

        with c_imp2:
            key_cls = f"{imp_crop}_classification"
            if key_cls in v2_weights:
                df_c = v2_weights[key_cls]
                m_family = df_c['Model_Family'].iloc[0]
                m_method = df_c['Method'].iloc[0]
                fig_imp_cls = create_feature_importance_bar(
                    df_c,
                    title=f"{imp_crop} V2 Risk Classification Weights ({m_family})"
                )
                st.plotly_chart(fig_imp_cls, use_container_width=True)
                st.caption(f"Method: {m_method} | Artifact: <code>risk_model_{imp_crop.lower()[:4]}.joblib</code>")
            else:
                st.info(f"V2 feature weights not applicable or unavailable for {imp_crop} classification.")

        # Safeguard 1: Provenance of Stage 3 Feature Importance
        with st.expander("📜 Historical Stage 3 Benchmark Permutation Importance (Pre-Cleansing Baseline)"):
            st.markdown(
                "**Provenance Disclosure (Safeguard 1):** The table below represents the permutation feature importance computed during "
                "the initial Stage 3 exploratory phase on the pre-cleansed dataset. It is provided strictly as a historical benchmark and "
                "must **never** be interpreted as the retrained V2 model parameter distribution."
            )
            v1_imp = load_v1_feature_importance_benchmark()
            if not v1_imp.empty:
                st.dataframe(v1_imp[v1_imp['Crop'] == imp_crop], use_container_width=True)

    with tab_status:
        st.subheader("Official Model Governance & Deployment Classifications")
        st.markdown(
            "In strict compliance with academic and professional engineering governance, models are categorized by verifiable generalization evidence:"
        )

        gov_records = [
            {
                'Crop': 'Rice',
                'Regression Status': 'Operational: Promising Retrospective',
                'Classification Status': 'Operational: Promising Retrospective',
                'Generalization Evidence': 'Positive test R² (+0.1963), ROC-AUC 0.9197, Recall 88.2%. Promising retrospective performance on 2010–2013 data.'
            },
            {
                'Crop': 'Groundnut',
                'Regression Status': 'Operational: Promising Retrospective',
                'Classification Status': 'Operational: Promising Retrospective',
                'Generalization Evidence': 'Positive test R² (+0.2103), ROC-AUC 0.9224, Recall 83.3%. Promising retrospective performance on 2010–2013 data.'
            },
            {
                'Crop': 'Maize',
                'Regression Status': 'Experimental Prototype',
                'Classification Status': 'Experimental Prototype',
                'Generalization Evidence': 'Regression R² restored (+0.0568) after bifurcation cleansing, but extreme post-2007 hybrid regime shift limits generalizability. Only 2 positive test risk cases.'
            },
            {
                'Crop': 'Cotton(lint)',
                'Regression Status': 'Research Prototype',
                'Classification Status': 'Research Prototype',
                'Generalization Evidence': 'Regression R² < 0 (-0.2394) due to rainfed variance. Classification evaluated on only 6 positive cases.'
            },
            {
                'Crop': 'Sugarcane',
                'Regression Status': 'Research Prototype',
                'Classification Status': 'Experimental Prototype',
                'Generalization Evidence': 'Regression R² (+0.0848) and Fallback restored (+0.0511) following 0.10x unit scaling. Classification discrimination remains near chance level (ROC-AUC = 0.6365).'
            }
        ]

        st.table(pd.DataFrame(gov_records))
        st.caption(
            "🛡️ **Deployment Disclosure (Safeguard 6):** Even for 'Operational' models (Rice, Groundnut), outputs reflect retrospective "
            "evaluation on historical data (2010–2013) and are not independently field-validated production advisory systems."
        )

