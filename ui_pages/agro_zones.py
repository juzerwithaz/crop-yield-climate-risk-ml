"""
agro_zones.py
Page 4: Agro-Zone Explorer for Tamil Nadu Crop Intelligence.
"""

import streamlit as st
import pandas as pd
from ui_components.cards import render_stat_card, render_callout, render_epistemic_badge
from ui_components.charts import create_district_marker_map, create_radar_comparison

def render_agro_zones_page(engine):
    st.markdown('<div class="badge-epistemic epistemic-association">[Agro-Climatic Typology]</div>', unsafe_allow_html=True)
    st.title("🗺️ Agro-Climatic Zone Explorer")
    st.markdown(
        "Interactive exploration of Tamil Nadu's 31 districts grouped into four coherent agro-climatic "
        "and soil clusters via K-Means (k=4) machine learning."
    )

    st.markdown("---")

    # Cluster Metadata
    c_meta = engine.cluster_metadata
    c_profiles = c_meta.get('cluster_profiles', {})
    c_names = c_meta.get('cluster_names', {})

    # Top KPI row: 4 Clusters
    col_c0, col_c1, col_c2, col_c3 = st.columns(4)
    with col_c0:
        render_stat_card(
            val=f"{c_profiles.get('0', {}).get('district_count', 13)} Dist.",
            label="Cluster 0",
            subtext=c_names.get('0', 'Central & Delta Semi-Arid Plains'),
            epistemic="Statistical Association"
        )
    with col_c1:
        render_stat_card(
            val=f"{c_profiles.get('1', {}).get('district_count', 4)} Dist.",
            label="Cluster 1",
            subtext=c_names.get('1', 'Western Ghats & Highland'),
            epistemic="Statistical Association"
        )
    with col_c2:
        render_stat_card(
            val=f"{c_profiles.get('2', {}).get('district_count', 3)} Dist.",
            label="Cluster 2",
            subtext=c_names.get('2', 'North-Western Plateau'),
            epistemic="Statistical Association"
        )
    with col_c3:
        render_stat_card(
            val=f"{c_profiles.get('3', {}).get('district_count', 11)} Dist.",
            label="Cluster 3",
            subtext=c_names.get('3', 'Coastal Humid Plain'),
            epistemic="Statistical Association"
        )

    # District Selector
    districts = engine.list_districts()
    sel_dist = st.selectbox(
        "Select District to Inspect Profile & Centroid Comparisons:",
        districts,
        index=districts.index("COIMBATORE") if "COIMBATORE" in districts else 0,
        key="agro_selected_district"
    )

    # Interactive Map & Radar Comparison
    m_col, r_col = st.columns([1.25, 1])

    with m_col:
        st.markdown("#### Geographic Coordinate Marker Map (Tamil Nadu)")
        map_fig = create_district_marker_map(c_meta, selected_district=sel_dist)
        st.plotly_chart(map_fig, use_container_width=True)
        st.caption(
            "📌 **Cartographic Disclosure (Safeguard 8):** Displays actual district coordinate markers "
            "derived from official project coordinates. Polygon boundaries are omitted to prevent artificial administrative fabrication."
        )

    with r_col:
        st.markdown("#### Selected District Environmental Profile")
        d_prof = engine.get_district_profile(sel_dist)
        cid = str(d_prof['cluster_id'])
        cname = d_prof['cluster_name']
        s_conf = d_prof['soil_confidence']
        d_feats = d_prof['district_historical_averages']
        c_means = d_prof['cluster_climate_soil_means']

        soil_conf_label = "Direct District Soil Survey (High Confidence)" if s_conf == 'real' else "Agro-Climatic Zone Average Estimate"
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size:1.2rem; font-weight:700; color:#1b4332;">{d_prof['district']}</div>
            <div style="font-size:0.875rem; color:#4b5563; margin-top:0.25rem;">
                <b>Administrative Zone:</b> {d_prof['zone']}<br/>
                <b>Cluster:</b> {cid} — {cname}<br/>
                <b>Soil Survey Provenance:</b> {soil_conf_label}<br/>
                <b>Coordinates:</b> {d_prof['latitude']:.4f}° N, {d_prof['longitude']:.4f}° E
            </div>
        </div>
        """, unsafe_allow_html=True)

        radar_fig = create_radar_comparison(
            district_name=sel_dist,
            district_features=d_feats,
            cluster_means=c_means,
            cluster_name=f"Cluster {cid}"
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    # Detailed Cluster Centroids Comparison Table
    st.markdown("### 📊 Agro-Climatic Cluster Centroids (Climate & Soil Baselines)")
    cluster_rows = []
    for c_id in sorted(c_profiles.keys(), key=lambda x: int(x)):
        p = c_profiles[c_id]
        means = p['climate_soil_means']
        cluster_rows.append({
            'Cluster ID': f"Cluster {c_id}",
            'Cluster Name': p['cluster_name'],
            'Districts Count': p['district_count'],
            'Mean Rainfall (mm)': f"{means['Rainfall']:.1f}",
            'Mean Temp (°C)': f"{means['Temp']:.1f}",
            'Mean Humidity (%)': f"{means['Humidity']:.1f}",
            'Mean pH': f"{means['pH']:.2f}",
            'Mean N (kg/ha)': f"{means['N']:.1f}",
            'Mean P (kg/ha)': f"{means['P']:.1f}",
            'Mean K (kg/ha)': f"{means['K']:.1f}",
            'Districts Included': ", ".join(p['districts'])
        })

    df_clusters = pd.DataFrame(cluster_rows)
    st.dataframe(df_clusters[['Cluster ID', 'Cluster Name', 'Districts Count', 'Mean Rainfall (mm)',
                             'Mean Temp (°C)', 'Mean Humidity (%)', 'Mean pH', 'Mean N (kg/ha)',
                             'Mean P (kg/ha)', 'Mean K (kg/ha)']], use_container_width=True)

    with st.expander("🔍 View Complete List of Districts in Each Cluster"):
        for row in cluster_rows:
            st.markdown(f"**{row['Cluster ID']} ({row['Cluster Name']}):** {row['Districts Included']}")

