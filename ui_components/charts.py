"""
charts.py
Interactive Plotly visualizations for the Tamil Nadu Crop Intelligence dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Agricultural Color Palette
PALETTE = {
    'primary': '#1b4332',
    'secondary': '#2d6a4f',
    'accent': '#52b788',
    'accent_light': '#74c69d',
    'lime': '#95d5b2',
    'forest': '#081c15',
    'charcoal': '#212529',
    'light_grey': '#f8f9fa',
    'warning': '#d97706',
    'danger': '#dc2626',
    'clusters': {
        0: '#2d6a4f',  # Central & Delta Semi-Arid Plains
        1: '#1d3557',  # Western Ghats & Highland
        2: '#e76f51',  # North-Western Plateau
        3: '#457b9d'   # Coastal Humid Plain
    }
}

def create_statewide_trends_chart(df: pd.DataFrame, selected_crop: str = None) -> go.Figure:
    """
    Creates an interactive historical yield trend chart across 1997-2013.
    """
    # Filter to eligible primary records
    clean_df = df[df['Data_Quality_Flag'].isin(['VALID', 'CLEANSED_UNIT_SCALED'])].copy()

    if selected_crop and selected_crop != "All Crops":
        clean_df = clean_df[clean_df['Crop'] == selected_crop]

    # Calculate annual weighted average yield (Production / Area)
    annual = clean_df.groupby(['Year', 'Crop']).apply(
        lambda g: g['Production'].sum() / g['Area'].sum() if g['Area'].sum() > 0 else np.nan
    ).reset_index(name='Statewide_Yield')

    fig = go.Figure()

    crops = sorted(annual['Crop'].unique())
    colors = ['#1b4332', '#2d6a4f', '#52b788', '#e76f51', '#1d3557']

    for i, crop in enumerate(crops):
        crop_data = annual[annual['Crop'] == crop]
        fig.add_trace(go.Scatter(
            x=crop_data['Year'],
            y=crop_data['Statewide_Yield'],
            mode='lines+markers',
            name=crop,
            line=dict(width=2.5, color=colors[i % len(colors)]),
            marker=dict(size=6),
            hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Yield: %{y:.2f} t/ha<extra></extra>"
        ))

    # Add vertical shading for temporal splits
    fig.add_vrect(x0=1996.5, x1=2007.5, fillcolor="#2d6a4f", opacity=0.06, line_width=0,
                  annotation_text="Training (<=2007)", annotation_position="top left")
    fig.add_vrect(x0=2007.5, x1=2009.5, fillcolor="#d97706", opacity=0.08, line_width=0,
                  annotation_text="Validation (2008-09)", annotation_position="top left")
    fig.add_vrect(x0=2009.5, x1=2013.5, fillcolor="#1d3557", opacity=0.06, line_width=0,
                  annotation_text="Test (2010-13)", annotation_position="top left")

    fig.update_layout(
        title="<b>Statewide Historical Yield Trends (1997–2013)</b><br><sup>[Observed Data] Area-weighted statewide yield across chronological splits</sup>",
        xaxis_title="Crop Year",
        yaxis_title="Yield (tonnes / hectare)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig

def create_soil_confidence_donut(cluster_metadata: dict) -> go.Figure:
    """
    Renders dynamic soil data confidence breakdown from V2 cluster metadata.
    Distinguishes direct district surveys from agro-climatic zone-average estimates.
    """
    district_assignments = cluster_metadata.get('district_assignments', {})

    counts = {'Direct District Survey (real)': 0, 'Agro-Climatic Zone Average (zone-avg)': 0, 'Low-Confidence Reference': 0}
    for d_name, d_info in district_assignments.items():
        sc = d_info.get('soil_confidence', 'zone-avg').lower()
        if 'real' in sc:
            counts['Direct District Survey (real)'] += 1
        elif 'zone' in sc:
            counts['Agro-Climatic Zone Average (zone-avg)'] += 1
        else:
            counts['Low-Confidence Reference'] += 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = ['#2d6a4f', '#74c69d', '#e9ecef']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#ffffff', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])

    fig.update_layout(
        title="<b>Soil Data Confidence Breakdown (31 Districts)</b><br><sup>[Observed Data] Dynamic classification from V2 cluster metadata</sup>",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def create_district_marker_map(cluster_metadata: dict, selected_district: str = None) -> go.Figure:
    """
    Safeguard 8: Clearly labelled district-coordinate marker map.
    Plots all 31 Tamil Nadu districts with cluster color encoding and coordinates.
    """
    assignments = cluster_metadata.get('district_assignments', {})
    cluster_names = cluster_metadata.get('cluster_names', {})

    records = []
    for dist, info in assignments.items():
        cid = info['cluster_id']
        cname = cluster_names.get(str(cid), f"Cluster {cid}")
        records.append({
            'District': dist,
            'Latitude': info['latitude'],
            'Longitude': info['longitude'],
            'Cluster_ID': cid,
            'Cluster_Name': cname,
            'Zone': info['zone'],
            'Soil_Confidence': info['soil_confidence'],
            'Rainfall': info['features']['Rainfall'],
            'Temp': info['features']['Temp'],
            'Is_Selected': (dist == selected_district)
        })

    map_df = pd.DataFrame(records)

    fig = go.Figure()

    # Plot markers grouped by cluster
    for cid in sorted(map_df['Cluster_ID'].unique()):
        c_sub = map_df[map_df['Cluster_ID'] == cid]
        c_name = c_sub['Cluster_Name'].iloc[0]
        c_color = PALETTE['clusters'].get(cid, '#2d6a4f')

        fig.add_trace(go.Scatter(
            x=c_sub['Longitude'],
            y=c_sub['Latitude'],
            mode='markers+text',
            name=f"C{cid}: {c_name}",
            text=c_sub['District'],
            textposition="top center",
            textfont=dict(size=9, color="#1f2937"),
            marker=dict(
                size=12,
                color=c_color,
                line=dict(width=1.5, color='#ffffff'),
                opacity=0.85
            ),
            customdata=np.stack((c_sub['Zone'], c_sub['Soil_Confidence'], c_sub['Rainfall'], c_sub['Temp']), axis=-1),
            hovertemplate=(
                "<b>District: %{text}</b><br>"
                "Cluster: " + c_name + "<br>"
                "Zone: %{customdata[0]}<br>"
                "Soil Confidence: %{customdata[1]}<br>"
                "Annual Rainfall: %{customdata[2]:.1f} mm<br>"
                "Mean Temperature: %{customdata[3]:.1f} °C<br>"
                "Coordinates: %{y:.3f}° N, %{x:.3f}° E<extra></extra>"
            )
        ))

    # Highlight selected district with a distinct halo marker if specified
    if selected_district and selected_district in assignments:
        sel_info = assignments[selected_district]
        fig.add_trace(go.Scatter(
            x=[sel_info['longitude']],
            y=[sel_info['latitude']],
            mode='markers',
            name=f"Selected: {selected_district}",
            marker=dict(
                size=22,
                color='rgba(255, 0, 0, 0)',
                line=dict(width=3, color='#dc2626')
            ),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title="<b>Tamil Nadu Agro-Climatic Zones & District Coordinate Markers</b><br><sup>[Statistical Association] K-Means (k=4) centroids mapped by geographic coordinates</sup>",
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        xaxis=dict(range=[76.0, 80.5], showgrid=True, gridcolor='#f3f4f6'),
        yaxis=dict(range=[7.8, 13.8], showgrid=True, gridcolor='#f3f4f6'),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=60, b=80),
        height=540
    )

    return fig

def create_radar_comparison(district_name: str, district_features: dict, cluster_means: dict, cluster_name: str) -> go.Figure:
    """
    Renders a radar chart comparing a district's climate and soil against its agro-zone cluster average.
    """
    categories = ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K']

    # Normalize values for clean radar visualization
    norm_ranges = {
        'Rainfall': (400, 1800),
        'Temp': (20, 32),
        'Humidity': (50, 85),
        'pH': (5.5, 8.5),
        'N': (100, 450),
        'P': (10, 60),
        'K': (100, 600)
    }

    d_norm = []
    c_norm = []
    for cat in categories:
        low, high = norm_ranges[cat]
        d_val = (district_features.get(cat, low) - low) / (high - low)
        c_val = (cluster_means.get(cat, low) - low) / (high - low)
        d_norm.append(np.clip(d_val, 0, 1))
        c_norm.append(np.clip(c_val, 0, 1))

    # Close the radar loop
    categories_loop = categories + [categories[0]]
    d_norm_loop = d_norm + [d_norm[0]]
    c_norm_loop = c_norm + [c_norm[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=d_norm_loop,
        theta=categories_loop,
        fill='toself',
        name=f"District: {district_name}",
        line=dict(color='#1b4332', width=2),
        fillcolor='rgba(27, 67, 50, 0.25)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=c_norm_loop,
        theta=categories_loop,
        fill='toself',
        name=f"Cluster Mean: {cluster_name}",
        line=dict(color='#e76f51', width=1.5, dash='dash'),
        fillcolor='rgba(231, 111, 81, 0.15)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)
        ),
        title=f"<b>Agro-Climatic Profile: {district_name} vs Cluster Centroid</b><br><sup>[Statistical Association] Standardized environmental & soil comparison</sup>",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

def create_regression_comparison_bar(reg_df: pd.DataFrame) -> go.Figure:
    """
    Renders comparative bar chart of Test R2 for V1 vs V2.
    Highlights gains and honestly displays Cotton deterioration.
    """
    # Filter to Primary Group B
    primary = reg_df[reg_df['Model_Type'] == 'Primary_Group_B'].copy()

    crops = primary['Crop'].tolist()
    r2_v1 = primary['Test_R2_V1'].tolist()
    r2_v2 = primary['Test_R2_V2'].tolist()

    fig = go.Figure(data=[
        go.Bar(name='V1 Baseline', x=crops, y=r2_v1, marker_color='#9ca3af'),
        go.Bar(name='V2 Cleansed', x=crops, y=r2_v2, marker_color='#2d6a4f')
    ])

    fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1)

    fig.update_layout(
        barmode='group',
        title="<b>Test Generalization R² Comparison (V1 Baseline vs V2 Cleansed)</b><br><sup>[Model Evaluation] Evaluated on protected test split (2010–2013)</sup>",
        xaxis_title="Crop",
        yaxis_title="Test R²",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig

def create_feature_importance_bar(df_imp: pd.DataFrame, title: str) -> go.Figure:
    """
    Renders horizontal bar chart for feature weights or importances.
    """
    fig = go.Figure(go.Bar(
        x=df_imp['Weight_Pct'],
        y=df_imp['Feature'],
        orientation='h',
        marker=dict(
            color=df_imp['Weight_Pct'],
            colorscale='Tealgrn',
            line=dict(color='#1b4332', width=1)
        ),
        hovertemplate="Feature: %{y}<br>Weight: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Relative Feature Weight / Importance (%)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig

def create_gauge_chart(value: float, min_val: float, max_val: float, title: str,
                       threshold_val: float = None, is_risk: bool = False, unit: str = "") -> go.Figure:
    """
    Renders a gauge chart for predicted yield or estimated risk probability.
    """
    bar_color = "#dc2626" if (is_risk and value >= (threshold_val or 0.5)) else "#2d6a4f"

    steps = []
    if is_risk and threshold_val is not None:
        steps = [
            {'range': [min_val, threshold_val], 'color': '#d1fae5'},
            {'range': [threshold_val, max_val], 'color': '#fee2e2'}
        ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': f" {unit}" if unit else "", 'font': {'size': 28, 'color': '#111827'}},
        title={'text': f"<b>{title}</b>", 'font': {'size': 15, 'color': '#374151'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "#4b5563"},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#e5e7eb",
            'steps': steps,
            'threshold': {
                'line': {'color': "#dc2626", 'width': 3},
                'thickness': 0.75,
                'value': threshold_val if threshold_val is not None else value
            } if threshold_val is not None else None
        }
    ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=25, r=25, t=50, b=25),
        height=240
    )

    return fig

