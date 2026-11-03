"""
inputs.py
Validated input widgets and parameter form builders for inference pages.
"""

import streamlit as st
import numpy as np

def render_scenario_input_form(engine, form_key: str = "prediction_form") -> dict:
    """
    Renders the agricultural input form with physical boundaries and district baseline autofill.
    Returns a dictionary of raw input parameters ready for validation.
    """
    col_crop, col_dist = st.columns([1, 1])

    with col_crop:
        crops = engine.list_crops()
        crop = st.selectbox(
            "Select Crop",
            crops,
            index=crops.index("Rice") if "Rice" in crops else 0,
            help="Five major Tamil Nadu crops supported by the V2 model pipelines.",
            key=f"{form_key}_crop"
        )

    with col_dist:
        districts = engine.list_districts()
        district = st.selectbox(
            "Select District",
            districts,
            index=districts.index("THANJAVUR") if "THANJAVUR" in districts else 0,
            help="31 Tamil Nadu administrative districts.",
            key=f"{form_key}_district"
        )

    # District Profile info
    d_prof = engine.get_district_profile(district)
    cid = d_prof['cluster_id']
    cname = d_prof['cluster_name']
    s_conf = d_prof['soil_confidence']
    d_feat = d_prof.get('district_historical_averages', {})
    c_means = d_prof.get('cluster_climate_soil_means', {})

    # Safeguard 7: If district baseline is missing or unavailable, fallback to cluster reference
    baseline = {}
    for feat in ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K']:
        if feat in d_feat and not np.isnan(d_feat[feat]):
            baseline[feat] = float(d_feat[feat])
        else:
            baseline[feat] = float(c_means.get(feat, 0.0))

    # Default typical crop lag values
    crop_default_lags = {
        'Rice': 3.4,
        'Groundnut': 1.8,
        'Maize': 2.5,
        'Cotton(lint)': 1.4,
        'Sugarcane': 105.0
    }
    def_lag = crop_default_lags.get(crop, 3.0)

    # District change detection
    last_district_key = f"{form_key}_last_district"
    last_district = st.session_state.get(last_district_key)
    district_changed = (last_district is not None) and (last_district != district)
    st.session_state[last_district_key] = district

    # Crop change detection
    last_crop_key = f"{form_key}_last_crop"
    last_crop = st.session_state.get(last_crop_key)
    crop_changed = (last_crop is not None) and (last_crop != crop)
    st.session_state[last_crop_key] = crop

    # Autofill helper button
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(
            f"<div style='font-size:0.85rem; color:#4b5563; padding: 0.35rem 0;'>"
            f"<b>Zone:</b> {d_prof['zone']} | <b>Cluster {cid}:</b> {cname} | "
            f"<b>Soil Confidence:</b> {'Direct District Survey' if s_conf == 'real' else 'Zone Average Estimate'}"
            f"</div>",
            unsafe_allow_html=True
        )

    with col_btn:
        autofill = st.button("Load Baseline Values", key=f"{form_key}_autofill_btn",
                             help=f"Populate inputs with historical baseline averages for {district}.")

    # Update or initialize session_state variables
    should_reload_baseline = autofill or district_changed or (f"{form_key}_rf_input" not in st.session_state)
    if should_reload_baseline:
        st.session_state[f"{form_key}_rf_input"] = float(round(baseline['Rainfall'], 1))
        st.session_state[f"{form_key}_temp_input"] = float(round(baseline['Temp'], 1))
        st.session_state[f"{form_key}_hum_input"] = float(round(baseline['Humidity'], 1))
        st.session_state[f"{form_key}_ph_input"] = float(round(baseline['pH'], 2))
        st.session_state[f"{form_key}_n_input"] = float(round(baseline['N'], 1))
        st.session_state[f"{form_key}_p_input"] = float(round(baseline['P'], 1))
        st.session_state[f"{form_key}_k_input"] = float(round(baseline['K'], 1))

    if autofill or crop_changed or (f"{form_key}_lag_input" not in st.session_state):
        st.session_state[f"{form_key}_lag_input"] = float(def_lag)

    if autofill:
        st.success(f"✅ Loaded historical baseline climate & soil values for {district}!")

    st.markdown("#### Climate Parameters")
    c1, c2, c3 = st.columns(3)

    with c1:
        rainfall = st.number_input(
            "Annual Rainfall (mm)",
            min_value=0.0,
            max_value=5000.0,
            value=st.session_state[f"{form_key}_rf_input"],
            step=25.0,
            help="Total annual rainfall (mm). Normal range: 500–1800 mm.",
            key=f"{form_key}_rf_input"
        )

    with c2:
        temp = st.number_input(
            "Mean Temperature (°C)",
            min_value=5.0,
            max_value=55.0,
            value=st.session_state[f"{form_key}_temp_input"],
            step=0.5,
            help="Average annual temperature (°C). Normal range: 23–30 °C.",
            key=f"{form_key}_temp_input"
        )

    with c3:
        humidity = st.number_input(
            "Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state[f"{form_key}_hum_input"],
            step=1.0,
            help="Average relative humidity (%). Normal range: 60–85%.",
            key=f"{form_key}_hum_input"
        )

    st.markdown("#### Soil Nutrient Parameters")
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        ph = st.number_input(
            "Soil pH",
            min_value=3.0,
            max_value=11.0,
            value=st.session_state[f"{form_key}_ph_input"],
            step=0.1,
            help="Soil acidity / alkalinity scale (3.0–11.0). Typical: 6.0–8.3.",
            key=f"{form_key}_ph_input"
        )

    with s2:
        n_val = st.number_input(
            "Available Nitrogen (N kg/ha)",
            min_value=0.0,
            max_value=1000.0,
            value=st.session_state[f"{form_key}_n_input"],
            step=10.0,
            help="Available Nitrogen in topsoil (kg/ha).",
            key=f"{form_key}_n_input"
        )

    with s3:
        p_val = st.number_input(
            "Available Phosphorus (P kg/ha)",
            min_value=0.0,
            max_value=300.0,
            value=st.session_state[f"{form_key}_p_input"],
            step=5.0,
            help="Available Phosphorus in topsoil (kg/ha).",
            key=f"{form_key}_p_input"
        )

    with s4:
        k_val = st.number_input(
            "Available Potassium (K kg/ha)",
            min_value=0.0,
            max_value=1000.0,
            value=st.session_state[f"{form_key}_k_input"],
            step=10.0,
            help="Available Potassium in topsoil (kg/ha).",
            key=f"{form_key}_k_input"
        )

    st.markdown("#### Historical Production Trend (Optional)")
    use_lag = st.checkbox(
        "Include 3-Year Historical Yield Lag (Yield_Lag_Avg)",
        value=True,
        help="If checked, uses Group B Primary model (significantly higher accuracy). If unchecked, uses Group A Fallback model.",
        key=f"{form_key}_use_lag"
    )

    yield_lag_avg = None
    if use_lag:
        yield_lag_avg = st.number_input(
            f"3-Year Historical Average Yield (t/ha) for {crop}",
            min_value=0.01,
            max_value=300.0,
            value=st.session_state[f"{form_key}_lag_input"],
            step=0.2 if crop != 'Sugarcane' else 5.0,
            help="Average yield of the preceding 3 years (t-1, t-2, t-3).",
            key=f"{form_key}_lag_input"
        )
    else:
        st.info("ℹ️ Mode: Climate-and-soil fallback model will be used. Historical yield trend is omitted.")

    return {
        'crop': crop,
        'district': district,
        'rainfall': rainfall,
        'temp': temp,
        'humidity': humidity,
        'ph': ph,
        'n': n_val,
        'p': p_val,
        'k': k_val,
        'yield_lag_avg': yield_lag_avg
    }
