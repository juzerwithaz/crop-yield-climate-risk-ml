"""
risk_analyzer.py
Page 3: Low-Yield Risk Analyzer for Tamil Nadu Crop Intelligence.
"""

import streamlit as st
from ui_components.inputs import render_scenario_input_form
from ui_components.cards import render_callout, render_stat_card, render_status_badge, render_epistemic_badge
from ui_components.charts import create_gauge_chart
from inference_v2 import ValidationError, InferenceError

def render_risk_analyzer_page(engine):
    st.markdown('<div class="badge-epistemic epistemic-prediction">[Risk Classification Engine]</div>', unsafe_allow_html=True)
    st.title("⚠️ Low-Yield Agricultural Risk Analyzer")
    st.markdown(
        "Evaluate the probabilistic risk of extreme low crop yield (falling into the historical bottom 20th percentile) "
        "under selected seasonal climate and soil conditions."
    )

    st.markdown("---")

    col_form, col_results = st.columns([1.15, 1])

    with col_form:
        st.subheader("1. Environmental & Management Inputs")
        inputs = render_scenario_input_form(engine, form_key="risk_analyzer")
        evaluate_clicked = st.button("Evaluate Low-Yield Risk", type="primary", use_container_width=True)

    with col_results:
        st.subheader("2. Risk Assessment & Decision Criteria")

        if evaluate_clicked:
            try:
                # Direct call to V2 classification engine
                res = engine.predict_risk(
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

                is_high_risk = res['is_high_risk']
                risk_class = res['predicted_risk_class']
                risk_prob = res['risk_probability']
                p_star = res['decision_threshold_p']
                tau20 = res['risk_threshold_tau20_t_ha']
                status_html = render_status_badge(res['model_status'], res['crop'])

                # 1. Primary Classification Result Card
                card_border = "#dc2626" if is_high_risk else "#2d6a4f"
                risk_badge_class = "badge-alert" if is_high_risk else "badge-operational"

                st.markdown(f"""
                <div class="stat-card" style="border-left: 5px solid {card_border};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="badge-epistemic epistemic-prediction">[Model Prediction]</span>
                        {status_html}
                    </div>
                    <div style="display:flex; align-items:baseline; margin-top:0.5rem; gap:12px;">
                        <span class="badge {risk_badge_class}" style="font-size:1.15rem; padding: 0.35rem 0.85rem;">{risk_class}</span>
                        <span style="font-size:1.4rem; font-weight:700; color:#1f2937;">P(Risk = 1) = {risk_prob:.2%}</span>
                    </div>
                    <div class="stat-label" style="margin-top:0.5rem;">Target Definition: Yield &le; {tau20:.2f} t/ha (Historical 20th Percentile &tau;<sub>20</sub>)</div>
                    <div class="stat-sub">Pipeline: <b>{res['model_family']}</b> | Decision Boundary (p*): <b>{p_star:.2f}</b> | Artifact: <code>{res['model_file']}</code></div>
                </div>
                """, unsafe_allow_html=True)

                # 2. Epistemic Disclosure on Uncalibrated Probability (Safeguard 4 & 5)
                render_callout(
                    title="⚠️ Model-Estimated & Uncalibrated Probability Disclosure",
                    text=(
                        f"The score <b>{risk_prob:.2%}</b> is a raw machine-learning output ({res['model_family']}). "
                        "It has <b>not</b> been calibrated via Platt scaling or isotonic regression on test data. "
                        "This classification serves as a relative risk prioritization indicator rather than a guaranteed "
                        "crop failure prediction or validated climate-disaster warning."
                    ),
                    level="warning"
                )

                # 3. Two-Threshold Explanation Card (Safeguard 4)
                st.markdown("#### ⚖️ Understanding the Two Thresholds")
                th_c1, th_c2 = st.columns(2)
                with th_c1:
                    render_stat_card(
                        val=f"{tau20:.2f} t/ha",
                        label="Yield-Risk Target Cutoff (τ₂₀)",
                        subtext=f"Historical 20th percentile of {res['crop']} training yields. Yields below this are labeled 'High Risk'.",
                        epistemic="Observed Data"
                    )
                with th_c2:
                    render_stat_card(
                        val=f"{p_star:.2f}",
                        label="Probability Decision Cutoff (p*)",
                        subtext="Model classification boundary. If estimated P(Risk) >= p*, flagged as 'High Risk'.",
                        epistemic="Model Prediction"
                    )

                # 4. Probability Gauge
                gauge_fig = create_gauge_chart(
                    value=risk_prob,
                    min_val=0.0,
                    max_val=1.0,
                    title="Estimated Low-Yield Probability P(Risk=1)",
                    threshold_val=p_star,
                    is_risk=True,
                    unit=""
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

                # 5. Out-of-Distribution Warnings
                if res['is_ood']:
                    for warn in res['ood_warnings']:
                        render_callout(
                            title="⚠️ Out-of-Distribution Input Notice",
                            text=warn,
                            level="warning"
                        )

                # 6. Fallback Notice
                if res['is_fallback']:
                    render_callout(
                        title="ℹ️ Climate-and-Soil Fallback Classifier",
                        text="Yield lag was not provided. Default decision threshold (0.50) is used with reduced discrimination.",
                        level="info"
                    )

                # 7. Model Status & Test Generalization
                st.markdown("#### Retrospective Test Performance (2010–2013)")
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.metric("Test ROC-AUC", f"{res['test_roc_auc']:.4f}")
                with r2:
                    st.metric("Test Recall", f"{res['test_recall']:.2%}")
                with r3:
                    st.metric("Test Precision", f"{res['test_precision']:.2%}")
                with r4:
                    st.metric("Test F1 Score", f"{res['test_f1']:.4f}")

                st.markdown(f"**Operational Rationale:** *{res['status_description']}*")
                st.caption(f"🛡️ **Disclosure:** {res['evaluation_disclosure']}")

            except ValidationError as ve:
                render_callout(
                    title="Input Limit Violation",
                    text=str(ve),
                    level="danger"
                )
            except InferenceError as ie:
                render_callout(
                    title="Inference Engine Error",
                    text=str(ie),
                    level="danger"
                )
        else:
            st.info("👈 Configure environmental conditions and click **Evaluate Low-Yield Risk**.")

