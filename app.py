"""
app.py
Main Application Entry Point for Tamil Nadu Crop Intelligence.
District-Level Crop Yield Prediction with Explainable Climate-Risk Clustering.
"""

import os
import streamlit as st
from utils.engine_provider import get_inference_engine
from ui_pages.overview import render_overview_page
from ui_pages.yield_predictor import render_yield_predictor_page
from ui_pages.risk_analyzer import render_risk_analyzer_page
from ui_pages.agro_zones import render_agro_zones_page
from ui_pages.explainability import render_explainability_page
from ui_pages.methodology import render_methodology_page

# Set Page Config
st.set_page_config(
    page_title="Tamil Nadu Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS Theme
def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

# Initialize Cached V2 Inference Engine
try:
    engine = get_inference_engine()
except Exception as e:
    st.error(f"Failed to load V2 Inference Engine: {str(e)}")
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌾 Tamil Nadu Crop Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-tagline">Understanding crop yield, climate patterns and agricultural risk across Tamil Nadu.</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    nav_options = [
        "1. Overview Dashboard",
        "2. Crop Yield Predictor",
        "3. Low-Yield Risk Analyzer",
        "4. Agro-Zone Explorer",
        "5. Explainability & Performance",
        "6. About & Methodology"
    ]

    selected_page = st.radio(
        "Navigation",
        nav_options,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🏷️ System Governance")
    st.markdown("""
    - **Engine:** `CropYieldInferenceEngineV2`
    - **Artifacts:** `model_artifacts_v2/`
    - **Dataset:** `V2 Cleansed (2,333 rows)`
    - **Validation:** Chronological split (<=2007 / 2008-09 / 2010-13)
    """)

    st.markdown("---")
    st.caption("Developed for rigorous academic research and explainable agricultural intelligence in Tamil Nadu.")

# Router
if selected_page == "1. Overview Dashboard":
    render_overview_page(engine)
elif selected_page == "2. Crop Yield Predictor":
    render_yield_predictor_page(engine)
elif selected_page == "3. Low-Yield Risk Analyzer":
    render_risk_analyzer_page(engine)
elif selected_page == "4. Agro-Zone Explorer":
    render_agro_zones_page(engine)
elif selected_page == "5. Explainability & Performance":
    render_explainability_page(engine)
elif selected_page == "6. About & Methodology":
    render_methodology_page()

