"""
yield_predictor.py
Page 2: Crop Yield Predictor for Tamil Nadu Crop Intelligence.
"""

import streamlit as st
from ui_components.inputs import render_scenario_input_form
from ui_components.cards import render_callout, render_stat_card, render_status_badge, render_epistemic_badge
from ui_components.charts import create_gauge_chart
from inference_v2 import ValidationError, InferenceError

def render_yield_predictor_page(engine):
    st.markdown('<div class="badge-epistemic epistemic-prediction">[Model Prediction Engine]</div>', unsafe_allow_html=True)
    st.title("🌱 Crop Yield Scenario Predictor")
    st.markdown(
        "Forecast expected crop yields under customized seasonal climate and soil conditions using the cleansed V2 models."
    )

    st.markdown("---")

    col_form, col_results = st.columns([1.15, 1])

    with col_form:
        st.subheader("1. Scenario Parameters")
        inputs = render_scenario_input_form(engine, form_key="yield_pred")
        predict_clicked = st.button("Generate Yield Prediction", type="primary", use_container_width=True)

    with col_results:
        st.subheader("2. Yield Forecast & Model Diagnostics")

        if predict_clicked:
            try:
                # Direct call to V2 inference engine
                res = engine.predict_yield(
                    crop=inputs['crop'],
                    district=inputs['district'],
                    rainfall=inputs['rainfall'],
                    temp=inputs['temp'],
                    humidity=inputs['humidity'],
                    ph=inputs['ph'],
                    n=inputs['n'],
                    p=inputs['p'],
                    k=inputs['k'],
                    yield_lag_avg=inputs['yield_lag_avg']
                )

                # 1. Primary Output Card
                pred_yield = res['predicted_yield']
                unit = res['unit']
                status_html = render_status_badge(res['model_status'], res['crop'])

                st.markdown(f"""
                <div class="stat-card" style="border-left: 5px solid #2d6a4f;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="badge-epistemic epistemic-prediction">[Model Prediction]</span>
                        {status_html}
                    </div>
                    <div class="stat-val" style="margin-top:0.5rem;">{pred_yield:.2f} <span style="font-size:1.1rem; color:#4b5563;">{unit}</span></div>
                    <div class="stat-label">Predicted Yield for {res['crop']} in {res['district']}</div>
                    <div class="stat-sub">Pipeline: <b>{res['model_family']}</b> ({res['feature_group_used']}) | Artifact: <code>{res['model_file']}</code></div>
                </div>
                """, unsafe_allow_html=True)

                # 2. Physical Floor / Clipping Warning
                if res['is_clipped']:
                    render_callout(
                        title="⚠️ Physical Floor Enforcement Applied",
                        text=res['clipping_warning'],
                        level="danger"
                    )

                # 3. Out-of-Distribution Warnings
                if res['is_ood']:
                    for warn in res['ood_warnings']:
                        if not res['is_clipped'] or "Physically invalid" not in warn:
                            render_callout(
                                title="⚠️ Input Distribution Advisory (OOD)",
                                text=warn,
                                level="warning"
                            )

                # 4. Fallback Model Notice
                if res['is_fallback']:
                    render_callout(
                        title="ℹ️ Climate-and-Soil Fallback Pipeline in Effect",
                        text=(
                            "Historical 3-year yield lag was not provided. The fallback model captures regional climate and "
                            "soil baseline potential, but has lower predictive accuracy without prior management history."
                        ),
                        level="info"
                    )

                # 5. Visual Gauge Comparison
                dist_prof = engine.get_district_profile(res['district'])
                max_gauge_val = 150.0 if res['crop'] == 'Sugarcane' else 8.0
                gauge_fig = create_gauge_chart(
                    value=pred_yield,
                    min_val=0.0,
                    max_val=max_gauge_val,
                    title=f"Predicted Yield ({unit})",
                    threshold_val=None,
                    is_risk=False,
                    unit=unit
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

                # 6. Model Governance & Status Rationale
                st.markdown("#### Model Operational Governance")
                st.markdown(f"**Operational Classification:** `{res['model_status']}`")
                st.markdown(f"*{res['status_description']}*")

                st.markdown("#### Test Split Validation Metrics (2010–2013)")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Test R²", f"{res['test_r2']:.4f}")
                with m2:
                    st.metric("Test RMSE", f"{res['test_rmse']:.2f} {unit}")
                with m3:
                    st.metric("Test MAE", f"{res['test_mae']:.2f} {unit}")

                st.caption(f"🛡️ **Scientific Disclosure:** {res['evaluation_disclosure']}")

            except ValidationError as ve:
                render_callout(
                    title="Input Validation Error (Physical Limit Violation)",
                    text=str(ve),
                    level="danger"
                )
            except InferenceError as ie:
                render_callout(
                    title="Inference Engine Failure",
                    text=str(ie),
                    level="danger"
                )
        else:
            st.info("👈 Configure scenario parameters and click **Generate Yield Prediction**.")

