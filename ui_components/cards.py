"""
cards.py
Reusable UI card components, status pills, and epistemic badges.
"""

import streamlit as st

def render_epistemic_badge(label_type: str):
    """
    Renders an epistemic classification tag:
    - 'Observed Data'
    - 'Model Prediction'
    - 'Statistical Association'
    - 'Agronomic Hypothesis'
    """
    css_class = {
        'Observed Data': 'epistemic-observed',
        'Model Prediction': 'epistemic-prediction',
        'Statistical Association': 'epistemic-association',
        'Agronomic Hypothesis': 'epistemic-hypothesis'
    }.get(label_type, 'epistemic-observed')

    html = f"""
    <div class="badge-epistemic {css_class}">[{label_type}]</div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_status_badge(status: str, crop_name: str = "") -> str:
    """Returns HTML for a standardized model governance status badge."""
    s_clean = status.lower()
    if 'operational' in s_clean:
        return '<span class="badge badge-operational">Operational: Promising Retrospective</span>'
    elif 'prototype' in s_clean:
        return '<span class="badge badge-prototype">Research Prototype</span>'
    elif 'experimental' in s_clean:
        return '<span class="badge badge-experimental">Experimental Prototype</span>'
    elif 'fallback' in s_clean:
        return '<span class="badge badge-neutral">Fallback (Climate + Soil)</span>'
    else:
        return f'<span class="badge badge-neutral">{status}</span>'

def render_stat_card(val: str, label: str, subtext: str = "", epistemic: str = None):
    """Renders a single KPI metric card."""
    epistemic_html = ""
    if epistemic:
        css = {
            'Observed Data': 'epistemic-observed',
            'Model Prediction': 'epistemic-prediction',
            'Statistical Association': 'epistemic-association',
            'Agronomic Hypothesis': 'epistemic-hypothesis'
        }.get(epistemic, 'epistemic-observed')
        epistemic_html = f'<div class="badge-epistemic {css}">[{epistemic}]</div>'

    html = f"""
    <div class="stat-card">
        {epistemic_html}
        <div class="stat-val">{val}</div>
        <div class="stat-label">{label}</div>
        {f'<div class="stat-sub">{subtext}</div>' if subtext else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_callout(text: str, level: str = "info", title: str = ""):
    """
    Renders a styled callout box.
    Levels: 'info', 'warning', 'danger'.
    """
    css = {
        'info': 'callout-info',
        'warning': 'callout-warning',
        'danger': 'callout-danger'
    }.get(level, 'callout-info')

    title_html = f"<strong>{title}</strong><br/>" if title else ""
    html = f"""
    <div class="custom-callout {css}">
        {title_html}{text}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

