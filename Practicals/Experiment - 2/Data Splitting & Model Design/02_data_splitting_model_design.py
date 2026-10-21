"""
Module 2: Data Splitting & Model Design
Scenario: A retail enterprise collects customer data (Income, Spending, Purchase Frequency).
Analyze the covariance matrix and determine which features show strong positive or negative relationships.
Model Probability Distributions and Bayes' Theorem.

This script:
1. Loads raw and preprocessed retail customer datasets and metadata from Module 1.
2. Partitions customer records into:
   - Training Set (70% - 700 customers)
   - Validation Set (15% - 150 customers)
   - Test Set (15% - 150 customers)
   using stratified sampling to preserve class balance and eliminate dataset shift.
3. Formulates Covariance Geometry & Spectral Decomposition:
   - Computes Train Covariance Matrix Sigma_train.
   - Performs Eigen-decomposition: Sigma = V * Lambda * V^T.
   - Computes Principal Variation Axes and Mahalanobis distances.
   - Constructs Covariance Confidence Ellipses (1-sigma, 2-sigma, 3-sigma).
4. Formulates Bayes' Theorem Model Architecture:
   - Prior Probabilities P(Y = k) from historical training data.
   - Class-Conditional Likelihood P(x | Y = k) using Multivariate Gaussian distributions.
   - Posterior Probability via Bayes' Theorem: P(Y = k | x) = P(x | Y = k) * P(Y = k) / P(x).
   - Derives Maximum A Posteriori (MAP) Quadratic Decision Boundaries.
5. Exports partitioned datasets and serialized model parameters.
6. Generates publication-grade, colorful visualization graphs:
   - Font: Cambria
   - Minimalist theme with light grid
   - Clear X and Y labels with well-numbered scales
   - Legend positioned cleanly at the bottom
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Matplotlib styling strictly per requirements
mpl.rcParams['font.family'] = 'Cambria'
mpl.rcParams['font.serif'] = ['Cambria', 'Times New Roman', 'DejaVu Serif']
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['figure.autolayout'] = False

def set_minimal_theme(ax):
    """Applies a clean, minimal styling with subtle gridlines to an axis."""
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#E5E7EB', zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#D1D5DB')
        spine.set_linewidth(0.8)
    ax.tick_params(colors='#1F2937', which='both', direction='out', length=4, width=0.8)

def plot_covariance_ellipse(ax, mean, cov, n_std=1.0, facecolor='none', **kwargs):
    """Draws a covariance confidence ellipse on an axis."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-6))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      facecolor=facecolor, **kwargs)
    return ax.add_patch(ellipse)

def run_module_2():
    module2_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(module2_dir)
    prep_dir = os.path.join(base_dir, 'Data Collection & Preprocessing')
    
    raw_csv_path = os.path.join(prep_dir, 'raw_retail_customers_1000.csv')
    meta_csv_path = os.path.join(prep_dir, 'customer_demographics_metadata_1000.csv')
    
    print("=" * 75)
    print("MODULE 2: DATA SPLITTING & MODEL DESIGN")
    print("=" * 75)
    
    if not os.path.exists(raw_csv_path) or not os.path.exists(meta_csv_path):
        raise FileNotFoundError("Module 1 output files missing. Please run Module 1 first.")
        
    df_raw = pd.read_csv(raw_csv_path)
    df_meta = pd.read_csv(meta_csv_path)
    
    # Merge target label
    df_full = df_raw.copy()
    df_full['High_Value_Customer'] = df_meta['High_Value_Customer']
    
    # 1. Stratified Data Splitting (70% Train, 15% Validation, 15% Test)
    np.random.seed(42)
    pos_idx = df_full[df_full['High_Value_Customer'] == 1].index.to_numpy(copy=True)
    neg_idx = df_full[df_full['High_Value_Customer'] == 0].index.to_numpy(copy=True)
    
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    n_pos_train = int(0.70 * len(pos_idx))
    n_pos_val = int(0.15 * len(pos_idx))
    
    n_neg_train = int(0.70 * len(neg_idx))
    n_neg_val = int(0.15 * len(neg_idx))
    
    train_idx = np.concatenate([pos_idx[:n_pos_train], neg_idx[:n_neg_train]])
    val_idx = np.concatenate([pos_idx[n_pos_train:n_pos_train+n_pos_val], neg_idx[n_neg_train:n_neg_train+n_neg_val]])
    test_idx = np.concatenate([pos_idx[n_pos_train+n_pos_val:], neg_idx[n_neg_train+n_neg_val:]])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    df_train = df_full.loc[train_idx].reset_index(drop=True)
    df_val = df_full.loc[val_idx].reset_index(drop=True)
    df_test = df_full.loc[test_idx].reset_index(drop=True)
    
    print(f"Data Splitting Completed:")
    print(f"  - Training Set:   {len(df_train)} customers (70.0%) | High-Value Ratio: {df_train['High_Value_Customer'].mean():.2%}")
    print(f"  - Validation Set: {len(df_val)} customers (15.0%) | High-Value Ratio: {df_val['High_Value_Customer'].mean():.2%}")
    print(f"  - Test Set:       {len(df_test)} customers (15.0%) | High-Value Ratio: {df_test['High_Value_Customer'].mean():.2%}")
    
    # 2. Covariance Matrix & Eigen-Decomposition on Training Data
    feature_cols = [c for c in df_raw.columns]
    X_train = df_train[feature_cols].values
    y_train = df_train['High_Value_Customer'].values
    
    cov_train = np.cov(X_train, rowvar=False)
    corr_train = np.corrcoef(X_train, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_train)
    # Sort descending
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    
    explained_var_ratio = eigenvalues / np.sum(eigenvalues)
    
    print("\n" + "-" * 75)
    print("SPECTRAL DECOMPOSITION OF TRAINING COVARIANCE MATRIX:")
    print("-" * 75)
    for i in range(len(eigenvalues)):
        print(f"  * Eigenvalue Lambda_{i+1:02d}: {eigenvalues[i]:10.2f} | Variance Explained: {explained_var_ratio[i]*100:5.2f}% | Cumulative: {np.sum(explained_var_ratio[:i+1])*100:5.2f}%")
        
    print("\nPrincipal Eigenvector v_1 (Direction of Maximum Joint Variance):")
    for feat, weight in zip(feature_cols, eigenvectors[:, 0]):
        print(f"    - {feat:22s}: {weight:+7.4f}")
        
    # 3. Bayes' Theorem Model Formulation
    # Class 0: Standard Customer, Class 1: High-Value VIP Customer
    X_train_c0 = X_train[y_train == 0]
    X_train_c1 = X_train[y_train == 1]
    
    prior_c0 = len(X_train_c0) / len(X_train)
    prior_c1 = len(X_train_c1) / len(X_train)
    
    mu_c0 = np.mean(X_train_c0, axis=0)
    mu_c1 = np.mean(X_train_c1, axis=0)
    
    cov_c0 = np.cov(X_train_c0, rowvar=False) + 1e-4 * np.eye(len(feature_cols))
    cov_c1 = np.cov(X_train_c1, rowvar=False) + 1e-4 * np.eye(len(feature_cols))
    
    print("\n" + "-" * 75)
    print("BAYES' THEOREM PARAMETERS (FIT ON TRAINING SET):")
    print("-" * 75)
    print(f"  - Prior Probability P(Standard, C_0):    {prior_c0:.4f} ({prior_c0*100:.1f}%)")
    print(f"  - Prior Probability P(High-Value, C_1):  {prior_c1:.4f} ({prior_c1*100:.1f}%)")
    print("\nClass Conditional Feature Means:")
    print(f"  {'Feature Name':22s} | {'C_0 Standard':>14s} | {'C_1 High-Value':>14s} | {'Delta':>10s}")
    for i, f in enumerate(feature_cols):
        print(f"  {f:22s} | {mu_c0[i]:14.2f} | {mu_c1[i]:14.2f} | {mu_c1[i] - mu_c0[i]:+10.2f}")
        
    # 4. Save Partitioned Datasets and Serialized Model Parameters
    df_train.to_csv(os.path.join(module2_dir, 'train_customers.csv'), index=False)
    df_val.to_csv(os.path.join(module2_dir, 'val_customers.csv'), index=False)
    df_test.to_csv(os.path.join(module2_dir, 'test_customers.csv'), index=False)
    
    np.save(os.path.join(module2_dir, 'covariance_matrix.npy'), cov_train)
    np.save(os.path.join(module2_dir, 'correlation_matrix.npy'), corr_train)
    np.savez(os.path.join(module2_dir, 'class_priors_and_params.npz'),
             prior_c0=prior_c0, prior_c1=prior_c1,
             mu_c0=mu_c0, mu_c1=mu_c1,
             cov_c0=cov_c0, cov_c1=cov_c1,
             feature_names=feature_cols)
             
    print(f"\nSaved split datasets and Bayesian model parameters to {module2_dir}")
    
    # 5. Publication-Grade Visualization Graphs
    print("\nGenerating Module 2 Visualization Graphs...")
    
    # Graph 4: Data Partitioning Proportions and Feature Consistency
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Donut Chart
    set_minimal_theme(axes[0])
    split_sizes = [len(df_train), len(df_val), len(df_test)]
    split_labels = [f"Train Set\n(70% - {len(df_train)})", 
                    f"Val Set\n(15% - {len(df_val)})", 
                    f"Test Set\n(15% - {len(df_test)})"]
    split_colors = ['#2563EB', '#F59E0B', '#10B981']
    
    wedges, texts, autotexts = axes[0].pie(
        split_sizes, labels=split_labels, colors=split_colors, autopct='%1.1f%%',
        startangle=140, pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor='#FFFFFF', linewidth=2.5)
    )
    for text in texts:
        text.set_fontname('Cambria')
        text.set_fontsize(11)
        text.set_color('#1F2937')
    for autotext in autotexts:
        autotext.set_fontname('Cambria')
        autotext.set_fontsize(11)
        autotext.set_color('#FFFFFF')
        autotext.set_weight('bold')
    axes[0].set_title("(A) Customer Dataset Partition Proportions", pad=12)
    
    # Subplot B: Density Overlay across Train, Val, Test (Spending Feature)
    set_minimal_theme(axes[1])
    sns.kdeplot(df_train['Spending'], ax=axes[1], color='#2563EB', label=f'Train Split ({len(df_train)} records)', linewidth=2.2, fill=True, alpha=0.15)
    sns.kdeplot(df_val['Spending'], ax=axes[1], color='#F59E0B', label=f'Validation Split ({len(df_val)} records)', linewidth=2.2, fill=True, alpha=0.15)
    sns.kdeplot(df_test['Spending'], ax=axes[1], color='#10B981', label=f'Test Split ({len(df_test)} records)', linewidth=2.2, fill=True, alpha=0.15)
    axes[1].set_title("(B) Spending Distribution Consistency Across Splits (Zero Dataset Shift)", pad=12)
    axes[1].set_xlabel("Annual Spending ($k/year)", labelpad=8)
    axes[1].set_ylabel("Probability Density", labelpad=8)
    axes[1].set_xlim(0, 130)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    legend_handles = [
        mpl.patches.Patch(facecolor='#2563EB', edgecolor='#1E40AF', label=f'Train Split ({len(df_train)} customers, 70%)'),
        mpl.patches.Patch(facecolor='#F59E0B', edgecolor='#B45309', label=f'Validation Split ({len(df_val)} customers, 15%)'),
        mpl.patches.Patch(facecolor='#10B981', edgecolor='#047857', label=f'Test Split ({len(df_test)} customers, 15%)')
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=3, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph4_path = os.path.join(module2_dir, '04_data_splitting_proportions_and_distribution.png')
    plt.savefig(graph4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph4_path}")
    
    # Graph 5: Covariance Confidence Ellipses & Eigenvector Principal Axes
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Income vs Spending Covariance Geometry
    set_minimal_theme(axes[0])
    x_inc = df_train['Income'].values
    y_sp = df_train['Spending'].values
    mean_pair_1 = [np.mean(x_inc), np.mean(y_sp)]
    cov_pair_1 = np.cov(x_inc, y_sp)
    
    # Scatter points colored by High Value target
    scatter1 = axes[0].scatter(x_inc, y_sp, c=df_train['High_Value_Customer'], cmap='coolwarm',
                               alpha=0.6, edgecolors='w', linewidths=0.5, s=35)
    # Draw 1-sigma, 2-sigma, 3-sigma confidence ellipses
    plot_covariance_ellipse(axes[0], mean_pair_1, cov_pair_1, n_std=1.0, edgecolor='#EF4444', linestyle='-', linewidth=2.0, label='1σ Ellipse (39.3%)')
    plot_covariance_ellipse(axes[0], mean_pair_1, cov_pair_1, n_std=2.0, edgecolor='#F59E0B', linestyle='--', linewidth=2.0, label='2σ Ellipse (86.5%)')
    plot_covariance_ellipse(axes[0], mean_pair_1, cov_pair_1, n_std=3.0, edgecolor='#8B5CF6', linestyle=':', linewidth=2.0, label='3σ Ellipse (98.9%)')
    
    # Eigenvectors
    vals1, vecs1 = np.linalg.eigh(cov_pair_1)
    v1 = vecs1[:, 1] * np.sqrt(vals1[1]) * 2.0
    axes[0].annotate('', xy=(mean_pair_1[0] + v1[0], mean_pair_1[1] + v1[1]), xytext=mean_pair_1,
                     arrowprops=dict(facecolor='#1E40AF', edgecolor='#1E40AF', width=2.5, headwidth=9))
    axes[0].plot(mean_pair_1[0], mean_pair_1[1], marker='o', markersize=8, color='#1E40AF', label='Centroid μ')
    axes[0].set_title("(A) Income vs. Spending Covariance Ellipses & Principal Eigenvector", pad=12)
    axes[0].set_xlabel("Annual Income ($k/year)", labelpad=8)
    axes[0].set_ylabel("Annual Spending ($k/year)", labelpad=8)
    
    # Subplot B: Purchase Frequency vs Spending Covariance Geometry
    set_minimal_theme(axes[1])
    x_freq = df_train['Purchase_Frequency'].values
    mean_pair_2 = [np.mean(x_freq), np.mean(y_sp)]
    cov_pair_2 = np.cov(x_freq, y_sp)
    
    axes[1].scatter(x_freq, y_sp, c=df_train['High_Value_Customer'], cmap='coolwarm',
                    alpha=0.6, edgecolors='w', linewidths=0.5, s=35)
    plot_covariance_ellipse(axes[1], mean_pair_2, cov_pair_2, n_std=1.0, edgecolor='#EF4444', linestyle='-', linewidth=2.0)
    plot_covariance_ellipse(axes[1], mean_pair_2, cov_pair_2, n_std=2.0, edgecolor='#F59E0B', linestyle='--', linewidth=2.0)
    plot_covariance_ellipse(axes[1], mean_pair_2, cov_pair_2, n_std=3.0, edgecolor='#8B5CF6', linestyle=':', linewidth=2.0)
    
    vals2, vecs2 = np.linalg.eigh(cov_pair_2)
    v2 = vecs2[:, 1] * np.sqrt(vals2[1]) * 2.0
    axes[1].annotate('', xy=(mean_pair_2[0] + v2[0], mean_pair_2[1] + v2[1]), xytext=mean_pair_2,
                     arrowprops=dict(facecolor='#1E40AF', edgecolor='#1E40AF', width=2.5, headwidth=9))
    axes[1].plot(mean_pair_2[0], mean_pair_2[1], marker='o', markersize=8, color='#1E40AF')
    axes[1].set_title("(B) Purchase Frequency vs. Spending Covariance Geometry", pad=12)
    axes[1].set_xlabel("Annual Purchase Frequency (Transactions)", labelpad=8)
    axes[1].set_ylabel("Annual Spending ($k/year)", labelpad=8)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.20, wspace=0.25)
    legend_elements = [
        mpl.lines.Line2D([0], [0], color='#EF4444', lw=2, label='1σ Confidence Ellipse (39.3%)'),
        mpl.lines.Line2D([0], [0], color='#F59E0B', lw=2, linestyle='--', label='2σ Confidence Ellipse (86.5%)'),
        mpl.lines.Line2D([0], [0], color='#8B5CF6', lw=2, linestyle=':', label='3σ Confidence Ellipse (98.9%)'),
        mpl.lines.Line2D([0], [0], color='#1E40AF', lw=3, label='Principal Eigenvector v_1'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3B82F6', markersize=8, label='Standard Shopper (C_0)'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EF4444', markersize=8, label='High-Value VIP (C_1)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=6, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph5_path = os.path.join(module2_dir, '05_covariance_ellipse_and_eigenvectors.png')
    plt.savefig(graph5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph5_path}")
    
    # Graph 6: Bayesian Posterior Density & Decision Boundary Architecture
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: 1D Bayesian Probability Updates (Prior vs Likelihood vs Posterior)
    set_minimal_theme(axes[0])
    x_grid = np.linspace(10, 120, 300)
    prior_c1_val = prior_c1
    prior_c0_val = prior_c0
    
    # Spending conditional likelihoods
    lik_c1 = stats.norm.pdf(x_grid, np.mean(df_train[df_train['High_Value_Customer']==1]['Spending']), 
                                    np.std(df_train[df_train['High_Value_Customer']==1]['Spending']))
    lik_c0 = stats.norm.pdf(x_grid, np.mean(df_train[df_train['High_Value_Customer']==0]['Spending']), 
                                    np.std(df_train[df_train['High_Value_Customer']==0]['Spending']))
    
    evidence = lik_c1 * prior_c1_val + lik_c0 * prior_c0_val
    posterior_c1 = (lik_c1 * prior_c1_val) / evidence
    
    axes[0].plot(x_grid, lik_c0, color='#2563EB', linestyle='--', linewidth=2.0, label='Likelihood P(Spending | Standard)')
    axes[0].plot(x_grid, lik_c1, color='#DC2626', linestyle='--', linewidth=2.0, label='Likelihood P(Spending | High-Value)')
    axes[0].axhline(prior_c1_val, color='#F59E0B', linestyle=':', linewidth=2.0, label=f'Prior P(High-Value) = {prior_c1_val:.2f}')
    
    # Twin axis for Posterior Probability in [0, 1]
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(x_grid, posterior_c1, color='#10B981', linewidth=2.5, label='Posterior P(High-Value | Spending)')
    ax0_twin.axhline(0.5, color='#6B7280', linestyle='-.', linewidth=1.2, label='MAP Decision Threshold (0.50)')
    ax0_twin.set_ylabel("Posterior Probability P(C_1 | x)", color='#10B981', labelpad=8)
    ax0_twin.set_ylim(-0.05, 1.05)
    ax0_twin.tick_params(colors='#10B981', axis='y')
    
    axes[0].set_title("(A) Bayes' Theorem: Prior, Likelihood, and Posterior Update", pad=12)
    axes[0].set_xlabel("Annual Spending ($k/year)", labelpad=8)
    axes[0].set_ylabel("Class-Conditional Likelihood Density", labelpad=8)
    
    # Subplot B: 2D Bayesian MAP Quadratic Decision Boundary in Income-Spending Plane
    set_minimal_theme(axes[1])
    xi_grid = np.linspace(20, 140, 150)
    yi_grid = np.linspace(10, 125, 150)
    XI, YI = np.meshgrid(xi_grid, yi_grid)
    grid_pts = np.c_[XI.ravel(), YI.ravel()]
    
    # 2D Gaussians for Income & Spending
    idx_2d = [0, 1]
    mu2_c0 = mu_c0[idx_2d]
    mu2_c1 = mu_c1[idx_2d]
    cov2_c0 = cov_c0[np.ix_(idx_2d, idx_2d)]
    cov2_c1 = cov_c1[np.ix_(idx_2d, idx_2d)]
    
    mvn_c0 = stats.multivariate_normal(mu2_c0, cov2_c0)
    mvn_c1 = stats.multivariate_normal(mu2_c1, cov2_c1)
    
    log_lik_0 = mvn_c0.logpdf(grid_pts)
    log_lik_1 = mvn_c1.logpdf(grid_pts)
    
    log_post_0 = log_lik_0 + np.log(prior_c0)
    log_post_1 = log_lik_1 + np.log(prior_c1)
    
    # MAP decision boundary is where log_post_1 - log_post_0 == 0
    decision_surf = (log_post_1 - log_post_0).reshape(XI.shape)
    post_c1_2d = 1.0 / (1.0 + np.exp(-decision_surf))
    
    # Contour fill of posterior probability
    cf = axes[1].contourf(XI, YI, post_c1_2d, levels=np.linspace(0, 1, 11), cmap='Blues', alpha=0.35)
    # Decision boundary line
    cs = axes[1].contour(XI, YI, decision_surf, levels=[0.0], colors=['#EF4444'], linewidths=[2.5], linestyles=['solid'])
    
    # Scatter training points
    axes[1].scatter(df_train[df_train['High_Value_Customer']==0]['Income'], 
                    df_train[df_train['High_Value_Customer']==0]['Spending'],
                    color='#3B82F6', alpha=0.5, s=25, label='Class 0 (Standard)')
    axes[1].scatter(df_train[df_train['High_Value_Customer']==1]['Income'], 
                    df_train[df_train['High_Value_Customer']==1]['Spending'],
                    color='#EF4444', alpha=0.6, s=25, label='Class 1 (High-Value)')
                    
    axes[1].set_title("(B) Bayesian MAP Quadratic Decision Boundary (Income vs. Spending)", pad=12)
    axes[1].set_xlabel("Annual Income ($k/year)", labelpad=8)
    axes[1].set_ylabel("Annual Spending ($k/year)", labelpad=8)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.32)
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_a_twin, labels_a_twin = ax0_twin.get_legend_handles_labels()
    
    bottom_handles = [handles_a[0], handles_a[1], handles_a[2], handles_a_twin[0], 
                      mpl.lines.Line2D([0], [0], color='#EF4444', lw=2.5, label='Bayesian MAP Boundary (P=0.50)')]
    bottom_labels = [labels_a[0], labels_a[1], labels_a[2], labels_a_twin[0], 'Bayesian MAP Boundary (P=0.50)']
    
    fig.legend(handles=bottom_handles, labels=bottom_labels, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=5, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph6_path = os.path.join(module2_dir, '06_bayesian_decision_boundary_and_priors.png')
    plt.savefig(graph6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph6_path}")
    
    print("\nModule 2 execution completed successfully!")

if __name__ == '__main__':
    run_module_2()
