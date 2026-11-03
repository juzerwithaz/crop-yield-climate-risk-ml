"""
11_stage2_district_clustering.py

Stage 2: District-Level Climate and Soil Clustering for Tamil Nadu (31 Districts).
- Deduplicates to one record per (District, Year) before computing long-term climate means.
- Merges with soil characteristics (pH, N, P, K, Soil_Confidence) and coordinates.
- Strictly excludes yield-derived variables and coordinates from clustering features.
- Evaluates K-Means, Agglomerative Hierarchical (Ward), and GMM across k=2 to 6.
- Assesses stability: seed variance, algorithm agreement (ARI), and low-confidence soil sensitivity (excluding Kanniyakumari and The Nilgiris).
- Generates comprehensive profiles, CSV outputs, and 5 diagnostic/spatial plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
)
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Paths
INPUT_STAGE1_CSV = "tn_master_dataset_stage1_ready.csv"
INPUT_SOIL_CSV = "tn_soil_data.csv"
INPUT_COORDS_CSV = "tn_district_coords.csv"
RESULTS_DIR = "stage2_results"

CLUSTERING_FEATURES = ['Rainfall', 'Temp', 'Humidity', 'pH', 'N', 'P', 'K']
RANDOM_SEED = 42

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def parse_coordinate(coord_str):
    """Parses coordinate strings like '11.1401° N' or '79.0747° E' into float."""
    if pd.isna(coord_str):
        return np.nan
    s = str(coord_str).replace('°', '').strip()
    parts = s.split()
    val = float(parts[0])
    if len(parts) > 1 and parts[1].upper() in ['S', 'W']:
        val = -val
    return val

def build_district_feature_matrix():
    print("Building district feature matrix...")
    df_s1 = pd.read_csv(INPUT_STAGE1_CSV)
    
    # 1. Deduplicate to exactly one record per (District, Year)
    weather_dy = df_s1[['District', 'Year', 'Rainfall', 'Temp', 'Humidity']].drop_duplicates()
    
    # Long-term climate mean per district across unique years
    climate_agg = weather_dy.groupby('District').agg({
        'Rainfall': ['mean', 'std'],
        'Temp': ['mean', 'std'],
        'Humidity': ['mean', 'std'],
        'Year': ['count', 'min', 'max']
    }).reset_index()
    
    climate_agg.columns = [
        'District', 'Rainfall', 'Rainfall_std', 'Temp', 'Temp_std', 
        'Humidity', 'Humidity_std', 'Years_Count', 'Year_Min', 'Year_Max'
    ]
    climate_agg['District_Norm'] = climate_agg['District'].str.strip().str.upper()
    
    # 2. Load Soil Data
    df_soil = pd.read_csv(INPUT_SOIL_CSV)
    df_soil['District_Norm'] = df_soil['District'].str.strip().str.upper()
    df_soil = df_soil.rename(columns={
        'N (kg/ha)': 'N', 'P (kg/ha)': 'P', 'K (kg/ha)': 'K', 'Confidence': 'Soil_Confidence'
    })
    
    # 3. Load District Coordinates
    df_coords = pd.read_csv(INPUT_COORDS_CSV)
    df_coords['District_Norm'] = df_coords['District_Name'].str.strip().str.upper()
    df_coords['Latitude_num'] = df_coords['Latitude'].apply(parse_coordinate)
    df_coords['Longitude_num'] = df_coords['Longitude'].apply(parse_coordinate)
    
    # 4. Merge all sources
    merged = pd.merge(climate_agg, df_soil[['District_Norm', 'Zone', 'pH', 'N', 'P', 'K', 'Soil_Confidence']], on='District_Norm')
    merged = pd.merge(merged, df_coords[['District_Norm', 'Latitude_num', 'Longitude_num']], on='District_Norm')
    
    # Order columns
    cols_order = [
        'District', 'Zone', 'Soil_Confidence', 'Latitude_num', 'Longitude_num',
        'Years_Count', 'Year_Min', 'Year_Max',
        'Rainfall', 'Rainfall_std', 'Temp', 'Temp_std', 'Humidity', 'Humidity_std',
        'pH', 'N', 'P', 'K'
    ]
    merged = merged[cols_order].sort_values('District').reset_index(drop=True)
    
    assert len(merged) == 31, f"Expected 31 districts, got {len(merged)}"
    assert not merged[CLUSTERING_FEATURES].isna().any().any(), "Found missing values in clustering features!"
    
    print(f"  Successfully assembled {len(merged)} districts.")
    print("  Soil Confidence distribution:")
    print(merged['Soil_Confidence'].value_counts().to_string())
    
    out_feat_csv = os.path.join(RESULTS_DIR, "stage2_district_features.csv")
    merged.to_csv(out_feat_csv, index=False)
    print(f"  Saved district feature matrix to: {out_feat_csv}")
    
    return merged

def evaluate_clustering_algorithms(X_scaled):
    print("\nEvaluating clustering algorithms across k=2 to 6...")
    records = []
    
    for k in range(2, 7):
        # 1. K-Means
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit(X_scaled)
        sil_km = silhouette_score(X_scaled, km.labels_)
        db_km = davies_bouldin_score(X_scaled, km.labels_)
        ch_km = calinski_harabasz_score(X_scaled, km.labels_)
        sizes_km = pd.Series(km.labels_).value_counts().to_dict()
        records.append({
            'Algorithm': 'K-Means', 'k': k, 'Silhouette': sil_km,
            'Davies_Bouldin': db_km, 'Calinski_Harabasz': ch_km,
            'Inertia': km.inertia_, 'BIC': np.nan, 'AIC': np.nan,
            'Cluster_Sizes': str(sizes_km)
        })
        
        # 2. Agglomerative (Ward Linkage)
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X_scaled)
        sil_agg = silhouette_score(X_scaled, agg.labels_)
        db_agg = davies_bouldin_score(X_scaled, agg.labels_)
        ch_agg = calinski_harabasz_score(X_scaled, agg.labels_)
        sizes_agg = pd.Series(agg.labels_).value_counts().to_dict()
        records.append({
            'Algorithm': 'Agglomerative_Ward', 'k': k, 'Silhouette': sil_agg,
            'Davies_Bouldin': db_agg, 'Calinski_Harabasz': ch_agg,
            'Inertia': np.nan, 'BIC': np.nan, 'AIC': np.nan,
            'Cluster_Sizes': str(sizes_agg)
        })
        
        # 3. Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_SEED, n_init=10).fit(X_scaled)
        gmm_labels = gmm.predict(X_scaled)
        if len(set(gmm_labels)) > 1:
            sil_gmm = silhouette_score(X_scaled, gmm_labels)
            db_gmm = davies_bouldin_score(X_scaled, gmm_labels)
            ch_gmm = calinski_harabasz_score(X_scaled, gmm_labels)
        else:
            sil_gmm, db_gmm, ch_gmm = np.nan, np.nan, np.nan
        sizes_gmm = pd.Series(gmm_labels).value_counts().to_dict()
        records.append({
            'Algorithm': 'GMM', 'k': k, 'Silhouette': sil_gmm,
            'Davies_Bouldin': db_gmm, 'Calinski_Harabasz': ch_gmm,
            'Inertia': np.nan, 'BIC': gmm.bic(X_scaled), 'AIC': gmm.aic(X_scaled),
            'Cluster_Sizes': str(sizes_gmm)
        })
        
    df_eval = pd.DataFrame(records)
    out_eval_csv = os.path.join(RESULTS_DIR, "stage2_clustering_comparison.csv")
    df_eval.to_csv(out_eval_csv, index=False)
    print(f"  Saved algorithm comparison to: {out_eval_csv}")
    return df_eval

def run_stability_and_sensitivity_analysis(df_dist, X_scaled, chosen_k=4):
    print(f"\nRunning stability and sensitivity analysis for k={chosen_k}...")
    stability_records = []
    
    # 1. Random seed stability for K-Means (10 seeds)
    base_km = KMeans(n_clusters=chosen_k, n_init=20, random_state=RANDOM_SEED).fit(X_scaled)
    base_labels = base_km.labels_
    
    seed_aris = []
    for seed in range(10):
        test_km = KMeans(n_clusters=chosen_k, n_init=20, random_state=seed).fit(X_scaled)
        ari = adjusted_rand_score(base_labels, test_km.labels_)
        seed_aris.append(ari)
        
    stability_records.append({
        'Test_Type': 'KMeans_Random_Seed_Sensitivity',
        'Description': 'Comparison across 10 random seeds (0-9) against seed 42',
        'Metric': 'Adjusted_Rand_Index (ARI)',
        'Mean_Value': float(np.mean(seed_aris)),
        'Min_Value': float(np.min(seed_aris)),
        'Max_Value': float(np.max(seed_aris)),
        'Notes': 'ARI = 1.0 indicates perfect cluster stability across initializations'
    })
    
    # 2. Algorithm Agreement: K-Means vs Agglomerative Ward
    ward_model = AgglomerativeClustering(n_clusters=chosen_k, linkage='ward').fit(X_scaled)
    ari_km_ward = adjusted_rand_score(base_labels, ward_model.labels_)
    stability_records.append({
        'Test_Type': 'Algorithm_Agreement',
        'Description': 'K-Means vs Agglomerative Ward Clustering on all 31 districts',
        'Metric': 'Adjusted_Rand_Index (ARI)',
        'Mean_Value': float(ari_km_ward),
        'Min_Value': float(ari_km_ward),
        'Max_Value': float(ari_km_ward),
        'Notes': 'Measures structural consensus between centroid-based and hierarchical linkage'
    })
    
    # 3. Low-Confidence Soil Sensitivity Analysis (Exclude Kanniyakumari and The Nilgiris)
    # The 2 districts are flagged 'low-confidence'
    low_conf_mask = df_dist['Soil_Confidence'] == 'low-confidence'
    excluded_names = df_dist[low_conf_mask]['District'].tolist()
    print(f"  Sensitivity test: excluding {len(excluded_names)} low-confidence districts: {excluded_names}")
    
    df_sens = df_dist[~low_conf_mask].copy().reset_index(drop=True)
    X_sens_raw = df_sens[CLUSTERING_FEATURES].values
    scaler_sens = StandardScaler()
    X_sens_scaled = scaler_sens.fit_transform(X_sens_raw)
    
    # Fit clustering on the 29 districts
    km_sens = KMeans(n_clusters=chosen_k, n_init=20, random_state=RANDOM_SEED).fit(X_sens_scaled)
    sens_labels = km_sens.labels_
    
    # Extract the labels of the 29 shared districts from the primary 31-district model
    primary_labels_for_shared = base_labels[~low_conf_mask]
    
    ari_soil_sens = adjusted_rand_score(primary_labels_for_shared, sens_labels)
    stability_records.append({
        'Test_Type': 'Soil_Uncertainty_Sensitivity',
        'Description': f'Excluding 2 low-confidence districts ({", ".join(excluded_names)}) and refitting on 29 districts',
        'Metric': 'Adjusted_Rand_Index (ARI)',
        'Mean_Value': float(ari_soil_sens),
        'Min_Value': float(ari_soil_sens),
        'Max_Value': float(ari_soil_sens),
        'Notes': 'Evaluated on the 29 shared districts (invariant to label permutations)'
    })
    
    df_stab = pd.DataFrame(stability_records)
    out_stab_csv = os.path.join(RESULTS_DIR, "stage2_cluster_stability.csv")
    df_stab.to_csv(out_stab_csv, index=False)
    print(f"  Saved stability and sensitivity report to: {out_stab_csv}")
    return df_stab

def generate_profiles_and_assignments(df_dist, X_scaled, chosen_k=4):
    print(f"\nGenerating cluster profiles and district assignments for k={chosen_k}...")
    km = KMeans(n_clusters=chosen_k, n_init=20, random_state=RANDOM_SEED).fit(X_scaled)
    labels = km.labels_
    
    # Meaningful cluster naming based on agro-climatic profile
    # Let's inspect cluster means to order them logically
    df_assigned = df_dist.copy()
    df_assigned['Cluster_ID'] = labels
    
    # Compute profiles in physical units
    profile_records = []
    for cid in range(chosen_k):
        sub = df_assigned[df_assigned['Cluster_ID'] == cid]
        dists = sub['District'].tolist()
        conf_dist = sub['Soil_Confidence'].value_counts().to_dict()
        
        row_prof = {
            'Cluster_ID': cid,
            'District_Count': len(sub),
            'Districts': "; ".join(dists),
            'Soil_Confidence_Breakdown': str(conf_dist)
        }
        for f in CLUSTERING_FEATURES:
            row_prof[f'{f}_mean'] = float(sub[f].mean())
            row_prof[f'{f}_std'] = float(sub[f].std())
            row_prof[f'{f}_median'] = float(sub[f].median())
            row_prof[f'{f}_min'] = float(sub[f].min())
            row_prof[f'{f}_max'] = float(sub[f].max())
            
        profile_records.append(row_prof)
        
    df_profiles = pd.DataFrame(profile_records)
    out_prof_csv = os.path.join(RESULTS_DIR, "stage2_cluster_profiles.csv")
    df_profiles.to_csv(out_prof_csv, index=False)
    print(f"  Saved cluster profiles to: {out_prof_csv}")
    
    # Save district assignments table
    assign_cols = [
        'District', 'Cluster_ID', 'Zone', 'Soil_Confidence',
        'Latitude_num', 'Longitude_num', 'Rainfall', 'Temp', 'Humidity',
        'pH', 'N', 'P', 'K'
    ]
    df_out_assign = df_assigned[assign_cols].sort_values(['Cluster_ID', 'District']).reset_index(drop=True)
    out_assign_csv = os.path.join(RESULTS_DIR, "stage2_cluster_assignments.csv")
    df_out_assign.to_csv(out_assign_csv, index=False)
    print(f"  Saved cluster assignments to: {out_assign_csv}")
    
    return df_assigned, df_profiles

def generate_visualizations(df_dist, X_scaled, df_eval, chosen_k=4):
    print("\nGenerating Stage 2 diagnostic and spatial plots...")
    
    # -------------------------------------------------------------
    # 1. Metric Comparison Plot across k (Silhouette, DB, Inertia)
    # -------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Silhouette
    for algo in ['K-Means', 'Agglomerative_Ward', 'GMM']:
        sub = df_eval[df_eval['Algorithm'] == algo]
        ax1.plot(sub['k'], sub['Silhouette'], marker='o', label=algo)
    ax1.axvline(chosen_k, color='grey', linestyle='--', alpha=0.7, label=f'Chosen k={chosen_k}')
    ax1.set_title('Silhouette Score (Higher is Better)')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=8)
    
    # Davies-Bouldin
    for algo in ['K-Means', 'Agglomerative_Ward', 'GMM']:
        sub = df_eval[df_eval['Algorithm'] == algo]
        ax2.plot(sub['k'], sub['Davies_Bouldin'], marker='s', label=algo)
    ax2.axvline(chosen_k, color='grey', linestyle='--', alpha=0.7)
    ax2.set_title('Davies-Bouldin Index (Lower is Better)')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Davies-Bouldin Index')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Elbow Inertia
    sub_km = df_eval[df_eval['Algorithm'] == 'K-Means']
    ax3.plot(sub_km['k'], sub_km['Inertia'], marker='^', color='#d62728', label='K-Means Inertia')
    ax3.axvline(chosen_k, color='grey', linestyle='--', alpha=0.7)
    ax3.set_title('K-Means Inertia Elbow Plot')
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Within-Cluster Inertia')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage2_metric_comparison.png"), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Hierarchical Dendrogram
    # -------------------------------------------------------------
    Z = linkage(X_scaled, method='ward')
    plt.figure(figsize=(11, 6))
    dendrogram(
        Z,
        labels=df_dist['District'].values,
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=Z[-(chosen_k - 1), 2]  # cut threshold for chosen_k
    )
    plt.title(f'Agglomerative Hierarchical Clustering Dendrogram (Ward Linkage, k={chosen_k})')
    plt.ylabel('Euclidean Distance in Standardized Feature Space')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage2_hierarchical_dendrogram.png"), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 3. PCA 2D Projection
    # -------------------------------------------------------------
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    km = KMeans(n_clusters=chosen_k, n_init=20, random_state=RANDOM_SEED).fit(X_scaled)
    labels = km.labels_
    
    plt.figure(figsize=(9, 6.5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=90, edgecolors='black', alpha=0.85)
    for i, txt in enumerate(df_dist['District']):
        plt.annotate(
            txt, (X_pca[i, 0], X_pca[i, 1]),
            fontsize=7, xytext=(4, 4), textcoords='offset points'
        )
    plt.xlabel(f'PCA Component 1 ({var_exp[0]*100:.1f}% variance)')
    plt.ylabel(f'PCA Component 2 ({var_exp[1]*100:.1f}% variance)')
    plt.title(f'District Climate-Soil Clusters: PCA Projection (k={chosen_k})\nTotal Explained Variance: {(var_exp[0]+var_exp[1])*100:.1f}%')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage2_pca_clusters.png"), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 4. Standardized Cluster Profile Heatmap
    # -------------------------------------------------------------
    df_std = pd.DataFrame(X_scaled, columns=CLUSTERING_FEATURES)
    df_std['Cluster_ID'] = labels
    cluster_means_std = df_std.groupby('Cluster_ID').mean()
    
    plt.figure(figsize=(8, 4.5))
    im = plt.imshow(cluster_means_std.values, cmap='coolwarm', aspect='auto', vmin=-2, vmax=2)
    plt.colorbar(im, label='Standardized Deviation from State Mean (Z-score)')
    plt.xticks(np.arange(len(CLUSTERING_FEATURES)), CLUSTERING_FEATURES, rotation=30)
    plt.yticks(np.arange(chosen_k), [f'Cluster {i}' for i in range(chosen_k)])
    plt.title(f'Cluster Feature Mean Profiles (Standardized Z-scores, k={chosen_k})')
    
    # Annotate values inside cells
    for r in range(chosen_k):
        for c in range(len(CLUSTERING_FEATURES)):
            val = cluster_means_std.values[r, c]
            color = 'black' if abs(val) < 1.0 else 'white'
            plt.text(c, r, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
            
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage2_cluster_heatmap.png"), dpi=200)
    plt.close()
    
    # -------------------------------------------------------------
    # 5. Spatial Geographic District Map
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 9))
    scatter_map = plt.scatter(
        df_dist['Longitude_num'], df_dist['Latitude_num'],
        c=labels, cmap='tab10', s=120, edgecolors='black', alpha=0.9
    )
    for i, txt in enumerate(df_dist['District']):
        plt.annotate(
            txt, (df_dist['Longitude_num'][i], df_dist['Latitude_num'][i]),
            fontsize=7, xytext=(4, 3), textcoords='offset points'
        )
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°N)')
    plt.title(f'Tamil Nadu District Agro-Climatic Clusters (Spatial Plot, k={chosen_k})')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stage2_spatial_map.png"), dpi=200)
    plt.close()
    
    print("  All 5 plots successfully saved to stage2_results/.")

def main():
    print("=" * 80)
    print("STAGE 2: DISTRICT-LEVEL CLIMATE AND SOIL CLUSTERING")
    print("=" * 80)
    
    ensure_dir(RESULTS_DIR)
    
    # 1. Build and save feature matrix
    df_dist = build_district_feature_matrix()
    
    # 2. Standardize features
    X_raw = df_dist[CLUSTERING_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 3. Evaluate algorithms across k=2 to 6
    df_eval = evaluate_clustering_algorithms(X_scaled)
    
    # Print summary of evaluation
    print("\nSummary of Clustering Comparison:")
    print(df_eval[['Algorithm', 'k', 'Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz', 'Inertia']].to_string(index=False))
    
    # 4. Final configuration selection
    # Multi-metric analysis: k=4 yields highest Silhouette for KMeans (0.347) and best Davies-Bouldin (0.972)
    chosen_k = 4
    print(f"\nSelected final clustering configuration: K-Means with k={chosen_k}")
    
    # 5. Stability & Sensitivity
    df_stab = run_stability_and_sensitivity_analysis(df_dist, X_scaled, chosen_k=chosen_k)
    print("\nStability Summary:")
    print(df_stab[['Test_Type', 'Metric', 'Mean_Value', 'Notes']].to_string(index=False))
    
    # 6. Generate profiles and assignments
    df_assigned, df_profiles = generate_profiles_and_assignments(df_dist, X_scaled, chosen_k=chosen_k)
    
    # 7. Generate diagnostic plots
    generate_visualizations(df_dist, X_scaled, df_eval, chosen_k=chosen_k)
    
    print("\n" + "=" * 80)
    print("STAGE 2 PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()

