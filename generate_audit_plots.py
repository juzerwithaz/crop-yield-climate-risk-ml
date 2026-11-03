"""
generate_audit_plots.py
Generates visual diagnostic figures for the ML Audit and Model Improvement report.
Saves all figures into ml_audit_results/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "ml_audit_results"

def plot_regression_comparison():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "stage1_test_metrics.csv"))
    
    crops = ['Cotton(lint)', 'Groundnut', 'Maize', 'Rice', 'Sugarcane']
    models = ['Primary_Group_B', 'Fallback_Group_A', 'Baseline_HistoricalYield', 'Baseline_DevMean']
    labels = ['Primary (Group B + Lag)', 'Fallback (Group A)', 'Historical Yield Persistence', 'Dev Mean Baseline']
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#7f7f7f']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Test R2
    x = np.arange(len(crops))
    width = 0.2
    
    for i, (m, lbl, col) in enumerate(zip(models, labels, colors)):
        sub = df[df['Model_Type'] == m].set_index('Crop').reindex(crops)
        r2_vals = sub['Test_R2'].values
        axes[0].bar(x + i*width - 1.5*width, r2_vals, width, label=lbl, color=col)
        
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_title("Test R² by Crop and Model Family (2010–2013)", fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(crops, fontsize=10)
    axes[0].set_ylabel("Test R²", fontsize=11)
    axes[0].set_ylim(-1.8, 0.35)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    axes[0].legend(fontsize=9, loc='lower left')
    
    # 2. Test RMSE
    for i, (m, lbl, col) in enumerate(zip(models, labels, colors)):
        sub = df[df['Model_Type'] == m].set_index('Crop').reindex(crops)
        # Exclude Sugarcane from direct RMSE scale or plot on log / secondary axis because its yield scale is ~100 t/ha
        rmse_vals = sub['Test_RMSE'].values
        # Normalize RMSE by crop training mean to make it comparative across crops!
        # Sugarcane mean is ~105, Rice is ~3.5, etc.
        # Alternatively plot crops except Sugarcane or normalized RMSE
        axes[1].bar(x + i*width - 1.5*width, rmse_vals, width, label=lbl, color=col)
        
    axes[1].set_title("Test RMSE by Crop and Model Family", fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(crops, fontsize=10)
    axes[1].set_ylabel("Test RMSE (t/ha)", fontsize=11)
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)
    axes[1].legend(fontsize=9, loc='upper left')
    
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "audit_regression_test_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

def plot_classification_comparison():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "stage3_test_metrics.csv"))
    crops = ['Cotton(lint)', 'Groundnut', 'Maize', 'Rice', 'Sugarcane']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. ROC-AUC
    models = ['Primary_Group_B', 'Fallback_Group_A', 'Baseline_DummyPrior']
    labels = ['Primary (Group B + Lag)', 'Fallback (Group A)', 'Dummy Prior Baseline']
    colors = ['#2ca02c', '#98df8a', '#7f7f7f']
    
    x = np.arange(len(crops))
    width = 0.25
    
    for i, (m, lbl, col) in enumerate(zip(models, labels, colors)):
        sub = df[df['Model_Type'] == m].set_index('Crop').reindex(crops)
        auc_vals = sub['ROC_AUC'].values
        axes[0].bar(x + i*width - width, auc_vals, width, label=lbl, color=col)
        
    axes[0].axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='Random Guessing (0.50)')
    axes[0].set_title("Test ROC-AUC Discrimination by Crop (2010–2013)", fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(crops, fontsize=10)
    axes[0].set_ylabel("ROC-AUC", fontsize=11)
    axes[0].set_ylim(0.4, 1.0)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    axes[0].legend(fontsize=9, loc='lower right')
    
    # 2. Recall and Balanced Accuracy for Primary Model
    prim = df[df['Model_Type'] == 'Primary_Group_B'].set_index('Crop').reindex(crops)
    rec_vals = prim['Recall'].values
    bacc_vals = prim['Balanced_Accuracy'].values
    f1_vals = prim['F1'].values
    
    axes[1].bar(x - 0.25, rec_vals, 0.25, label='Recall (Risk Sensitivity)', color='#d62728')
    axes[1].bar(x, bacc_vals, 0.25, label='Balanced Accuracy', color='#ff7f0e')
    axes[1].bar(x + 0.25, f1_vals, 0.25, label='F1 Score', color='#1f77b4')
    
    axes[1].set_title("Primary Model Test Risk Warning Metrics", fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(crops, fontsize=10)
    axes[1].set_ylabel("Metric Value", fontsize=11)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)
    axes[1].legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "audit_classification_test_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

def plot_threshold_sensitivity():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "stage3_validation_comparison.csv"))
    
    # Filter for selected models
    selected_map = {
        'Cotton(lint)': 'RandomForest',
        'Groundnut': 'LogisticRegression',
        'Rice': 'RandomForest',
        'Sugarcane': 'HistGradientBoosting'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (crop, mod) in enumerate(selected_map.items()):
        ax = axes[i]
        sub = df[(df['Crop'] == crop) & (df['Model'] == mod) & (df['FeatureGroup'] == 'Group_B_Lag')]
        sub = sub.sort_values('Prob_Threshold')
        
        ax.plot(sub['Prob_Threshold'], sub['F1'], marker='o', linewidth=2, color='#1f77b4', label='Validation F1')
        ax.plot(sub['Prob_Threshold'], sub['Recall'], marker='s', linewidth=1.5, linestyle='--', color='#d62728', label='Recall')
        ax.plot(sub['Prob_Threshold'], sub['Precision'], marker='^', linewidth=1.5, linestyle=':', color='#2ca02c', label='Precision')
        
        ax.axvline(0.50, color='gray', linestyle='--', alpha=0.7, label='Default Threshold (0.50)')
        ax.set_title(f"{crop} ({mod})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Decision Probability Threshold (p*)", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(fontsize=8, loc='best')
            
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "audit_threshold_tuning_curves.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    plot_regression_comparison()
    plot_classification_comparison()
    plot_threshold_sensitivity()

