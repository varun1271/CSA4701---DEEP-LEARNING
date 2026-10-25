"""
Module 2: Data Splitting & Model Design
Scenario: A manufacturing unit records the number of defective products in different batches.
Model the defect counts using a suitable probability distribution and estimate the probability
of observing a specified number of defects. Analyze Entropy, Cross-Entropy, and KL Divergence.

This script:
1. Loads raw and preprocessed manufacturing batch datasets from Module 1.
2. Partitions batches into:
   - Training Set (70% - 700 batches)
   - Validation Set (15% - 150 batches)
   - Test Set (15% - 150 batches)
   using stratified sampling to preserve defect rate distributions and avoid dataset shift.
3. Formulates Defect Probability Estimation Models:
   - Evaluates PMF P(K = k) and CDF F(k) = P(K <= k).
   - Estimates probabilities for critical operational defect thresholds:
     k = 0 (Pristine zero-defect batch), k = 3 (Normal target), k = 7 (Warning), k >= 10 (Critical rejection).
4. Explores Information Theory Geometry & Convexity:
   - Proves mathematically that Maximum Likelihood Estimation (MLE) minimizes KL Divergence:
     arg min_theta D_KL(P || Q_theta) == arg max_theta sum log Q(k_i | theta).
   - Evaluates the convex profile of Cross-Entropy H(P, Q_theta) and KL Divergence D_KL(P || Q_theta)
     across parameter spaces (Poisson lambda and Negative Binomial r, p).
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

def compute_kl_divergence_vector(p_emp, q_model, eps=1e-12):
    """Vectorized calculation of H(P), H(P, Q), and D_KL(P || Q)."""
    p = np.clip(p_emp, eps, 1.0)
    p = p / np.sum(p)
    q = np.clip(q_model, eps, 1.0)
    q = q / np.sum(q)
    h_p = -np.sum(p * np.log2(p))
    cross_ent = -np.sum(p * np.log2(q))
    kl_div = max(cross_ent - h_p, 0.0)
    return h_p, cross_ent, kl_div

def run_module_2():
    module2_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(module2_dir)
    prep_dir = os.path.join(base_dir, 'Data Collection & Preprocessing')
    
    raw_csv_path = os.path.join(prep_dir, 'raw_manufacturing_batches_1000.csv')
    meta_csv_path = os.path.join(prep_dir, 'batch_metadata_and_inspection_1000.csv')
    
    print("=" * 75)
    print("MODULE 2: DATA SPLITTING & MODEL DESIGN (INFORMATION THEORY & ESTIMATION)")
    print("=" * 75)
    
    if not os.path.exists(raw_csv_path) or not os.path.exists(meta_csv_path):
        raise FileNotFoundError("Module 1 output files missing. Please run Module 1 first.")
        
    df_raw = pd.read_csv(raw_csv_path)
    df_meta = pd.read_csv(meta_csv_path)
    
    df_full = df_raw.copy()
    df_full['Is_Rejected_Batch'] = df_meta['Is_Rejected_Batch']
    df_full['Quality_Status'] = df_meta['Quality_Status']
    
    # 1. Stratified Data Splitting (70% Train, 15% Validation, 15% Test)
    np.random.seed(42)
    rej_idx = df_full[df_full['Is_Rejected_Batch'] == 1].index.to_numpy(copy=True)
    acc_idx = df_full[df_full['Is_Rejected_Batch'] == 0].index.to_numpy(copy=True)
    
    np.random.shuffle(rej_idx)
    np.random.shuffle(acc_idx)
    
    n_rej_train = int(0.70 * len(rej_idx))
    n_rej_val = int(0.15 * len(rej_idx))
    
    n_acc_train = int(0.70 * len(acc_idx))
    n_acc_val = int(0.15 * len(acc_idx))
    
    train_idx = np.concatenate([rej_idx[:n_rej_train], acc_idx[:n_acc_train]])
    val_idx = np.concatenate([rej_idx[n_rej_train:n_rej_train+n_rej_val], acc_idx[n_acc_train:n_acc_train+n_acc_val]])
    test_idx = np.concatenate([rej_idx[n_rej_train+n_rej_val:], acc_idx[n_acc_train+n_acc_val:]])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    df_train = df_full.loc[train_idx].reset_index(drop=True)
    df_val = df_full.loc[val_idx].reset_index(drop=True)
    df_test = df_full.loc[test_idx].reset_index(drop=True)
    
    print(f"Data Splitting Completed (Stratified by Batch Rejection):")
    print(f"  - Training Set:   {len(df_train)} batches (70.0%) | Critical Scrap Rate: {df_train['Is_Rejected_Batch'].mean():.2%}")
    print(f"  - Validation Set: {len(df_val)} batches (15.0%) | Critical Scrap Rate: {df_val['Is_Rejected_Batch'].mean():.2%}")
    print(f"  - Test Set:       {len(df_test)} batches (15.0%) | Critical Scrap Rate: {df_test['Is_Rejected_Batch'].mean():.2%}")
    
    # 2. Fit Probability Models Exclusively on Training Split
    defects_train = df_train['Defect_Count'].values
    mean_tr = np.mean(defects_train)
    var_tr = np.var(defects_train, ddof=1)
    
    # Negative Binomial Method of Moments
    p_nb_tr = mean_tr / var_tr
    r_nb_tr = (mean_tr ** 2) / (var_tr - mean_tr)
    
    # Poisson
    lambda_pois_tr = mean_tr
    
    # Binomial (n = 500)
    p_binom_tr = mean_tr / 500.0
    
    print("\n" + "-" * 75)
    print("TRAINING FIT PARAMETERS & PROBABILITY ESTIMATION ARCHITECTURE:")
    print("-" * 75)
    print(f"  * Training Mean Defect Count:           {mean_tr:.4f}")
    print(f"  * Training Defect Variance:             {var_tr:.4f} (Overdispersion index: {var_tr/mean_tr:.2f})")
    print(f"  * Negative Binomial Model:              NB(r = {r_nb_tr:.4f}, p = {p_nb_tr:.4f})")
    print(f"  * Poisson Model:                        Pois(lambda = {lambda_pois_tr:.4f})")
    print(f"  * Binomial Model:                       Bin(n = 500, p = {p_binom_tr:.6f})")
    
    # 3. Probability of Observing Specified Numbers of Defects (Theoretical vs Empirical)
    k_targets = [0, 1, 2, 3, 5, 7, 10, 15]
    print("\nPROBABILITY OF OBSERVING SPECIFIED DEFECT COUNTS P(K = k):")
    print(f"  {'Defect Count (k)':18s} | {'Empirical P(k)':>16s} | {'Negative Binomial':>18s} | {'Poisson Model':>15s} | {'Compliance Risk':>16s}")
    print("  " + "-" * 90)
    for k in k_targets:
        emp_p = np.mean(defects_train == k)
        nb_p = stats.nbinom.pmf(k, r_nb_tr, p_nb_tr)
        pois_p = stats.poisson.pmf(k, lambda_pois_tr)
        
        if k == 0:
            desc = "Zero-Defect Run"
        elif k <= 3:
            desc = "Target Optimal"
        elif k <= 7:
            desc = "Warning Band"
        else:
            desc = "Critical Scrap"
            
        print(f"  k = {k:2d} defects      | {emp_p*100:15.2f}% | {nb_p*100:17.2f}% | {pois_p*100:14.2f}% | {desc:>16s}")
        
    # Cumulative Rejection Risk P(K >= 10)
    emp_scrap = np.mean(defects_train >= 10)
    nb_scrap = 1.0 - stats.nbinom.cdf(9, r_nb_tr, p_nb_tr)
    pois_scrap = 1.0 - stats.poisson.cdf(9, lambda_pois_tr)
    print("  " + "-" * 90)
    print(f"  Critical Scrap Risk P(K >= 10): Empirical = {emp_scrap*100:.2f}% | NB Model = {nb_scrap*100:.2f}% | Poisson Model = {pois_scrap*100:.2f}%")
    print(f"  Notice: Poisson underestimates critical batch scrap risk by {(nb_scrap - pois_scrap)*100:.2f}% due to unmodeled variance!")
    
    # 4. Save Partitioned Datasets and Fitted Parameters
    df_train.to_csv(os.path.join(module2_dir, 'train_batches.csv'), index=False)
    df_val.to_csv(os.path.join(module2_dir, 'val_batches.csv'), index=False)
    df_test.to_csv(os.path.join(module2_dir, 'test_batches.csv'), index=False)
    
    np.savez(os.path.join(module2_dir, 'fitted_distribution_params.npz'),
             r_nb=r_nb_tr, p_nb=p_nb_tr,
             lambda_pois=lambda_pois_tr,
             p_binom=p_binom_tr,
             mean_train=mean_tr, var_train=var_tr)
             
    print(f"\nSaved split datasets and serialized parameters to {module2_dir}")
    
    # 5. Publication-Grade Visualization Graphs
    print("\nGenerating Module 2 Visualization Graphs...")
    
    # Graph 4: Data Splitting Proportions & Defect Consistency
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
    axes[0].set_title("(A) Manufacturing Batch Partition Proportions", pad=12)
    
    # Subplot B: Density Overlay across Train, Val, Test (Defect Count)
    set_minimal_theme(axes[1])
    sns.kdeplot(df_train['Defect_Count'], ax=axes[1], color='#2563EB', label=f'Train Split ({len(df_train)} batches)', linewidth=2.2, fill=True, alpha=0.15)
    sns.kdeplot(df_val['Defect_Count'], ax=axes[1], color='#F59E0B', label=f'Validation Split ({len(df_val)} batches)', linewidth=2.2, fill=True, alpha=0.15)
    sns.kdeplot(df_test['Defect_Count'], ax=axes[1], color='#10B981', label=f'Test Split ({len(df_test)} batches)', linewidth=2.2, fill=True, alpha=0.15)
    axes[1].set_title("(B) Defect Count Distribution Consistency Across Splits (Zero Dataset Shift)", pad=12)
    axes[1].set_xlabel("Defect Count per Batch (k)", labelpad=8)
    axes[1].set_ylabel("Probability Density", labelpad=8)
    axes[1].set_xlim(-1, 24)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    legend_handles = [
        mpl.patches.Patch(facecolor='#2563EB', edgecolor='#1E40AF', label=f'Train Split ({len(df_train)} batches, 70%)'),
        mpl.patches.Patch(facecolor='#F59E0B', edgecolor='#B45309', label=f'Validation Split ({len(df_val)} batches, 15%)'),
        mpl.patches.Patch(facecolor='#10B981', edgecolor='#047857', label=f'Test Split ({len(df_test)} batches, 15%)')
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=3, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph4_path = os.path.join(module2_dir, '04_data_splitting_proportions_and_distribution.png')
    plt.savefig(graph4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph4_path}")
    
    # Graph 5: PMF and Cumulative Distribution Function (CDF) with Quality Control Thresholds
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
    fig.patch.set_facecolor('#FFFFFF')
    
    k_eval = np.arange(0, 22)
    
    # Subplot A: PMF Probability Mass Function
    set_minimal_theme(axes[0])
    p_emp_tr = np.array([np.mean(defects_train == k) for k in k_eval])
    axes[0].bar(k_eval, p_emp_tr, color='#93C5FD', edgecolor='#1E40AF', alpha=0.55, width=0.7, label='Empirical Defect Mass P(k)')
    axes[0].plot(k_eval, stats.nbinom.pmf(k_eval, r_nb_tr, p_nb_tr), marker='o', color='#10B981', linewidth=2.4, label='Neg-Binomial PMF (Optimal)')
    axes[0].plot(k_eval, stats.poisson.pmf(k_eval, lambda_pois_tr), marker='s', color='#EF4444', linestyle='--', linewidth=2.0, label='Poisson PMF (Underdispersed)')
    axes[0].axvline(3.5, color='#10B981', linestyle=':', linewidth=1.5, label='Target Optimal Line (k <= 3)')
    axes[0].axvline(9.5, color='#EF4444', linestyle=':', linewidth=1.5, label='Critical Scrap Threshold (k >= 10)')
    axes[0].set_title("(A) Defect Probability Mass Function P(K = k)", pad=12)
    axes[0].set_xlabel("Number of Defective Products per Batch (k)", labelpad=8)
    axes[0].set_ylabel("Probability P(K = k)", labelpad=8)
    axes[0].set_xlim(-0.5, 21.5)
    
    # Subplot B: Cumulative Distribution Function F(k) = P(K <= k)
    set_minimal_theme(axes[1])
    cdf_emp = np.cumsum(p_emp_tr)
    cdf_nb = stats.nbinom.cdf(k_eval, r_nb_tr, p_nb_tr)
    cdf_pois = stats.poisson.cdf(k_eval, lambda_pois_tr)
    
    axes[1].step(k_eval, cdf_emp, where='post', color='#2563EB', linewidth=2.4, label='Empirical Cumulative CDF')
    axes[1].plot(k_eval, cdf_nb, marker='o', color='#10B981', linewidth=2.2, label='Neg-Binomial CDF')
    axes[1].plot(k_eval, cdf_pois, marker='s', color='#EF4444', linestyle='--', linewidth=1.8, label='Poisson CDF')
    axes[1].axhline(0.95, color='#F59E0B', linestyle='--', linewidth=1.5, label='95% Quality Compliance Bound')
    axes[1].axvline(9.5, color='#EF4444', linestyle=':', linewidth=1.5)
    axes[1].set_title("(B) Cumulative Defect Distribution Function F(k) = P(K ≤ k)", pad=12)
    axes[1].set_xlabel("Cumulative Defect Threshold (k)", labelpad=8)
    axes[1].set_ylabel("Cumulative Probability P(K ≤ k)", labelpad=8)
    axes[1].set_xlim(-0.5, 21.5)
    axes[1].set_ylim(-0.02, 1.05)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    handles5_a, labels5_a = axes[0].get_legend_handles_labels()
    handles5_b, labels5_b = axes[1].get_legend_handles_labels()
    all_h5 = [handles5_a[0], handles5_a[1], handles5_a[2], handles5_b[3], handles5_a[4]]
    all_l5 = [labels5_a[0], labels5_a[1], labels5_a[2], '95% Compliance Bound', 'Critical Scrap Threshold (k=10)']
    fig.legend(handles=all_h5, labels=all_l5, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=5, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph5_path = os.path.join(module2_dir, '05_theoretical_vs_empirical_pmf_and_cdf.png')
    plt.savefig(graph5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph5_path}")
    
    # Graph 6: Information Theory Surfaces: Entropy, Cross-Entropy & Convexity of KL Divergence
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Convex Profile of D_KL and Cross-Entropy across Poisson parameter lambda
    set_minimal_theme(axes[0])
    lambda_grid = np.linspace(1.5, 12.0, 150)
    kl_vals = []
    ce_vals = []
    k_full = np.arange(0, 26)
    p_true = np.array([np.mean(defects_train == k) for k in k_full])
    h_true, _, _ = compute_kl_divergence_vector(p_true, p_true)
    
    for l_cand in lambda_grid:
        q_cand = stats.poisson.pmf(k_full, mu=l_cand)
        _, ce_l, kl_l = compute_kl_divergence_vector(p_true, q_cand)
        kl_vals.append(kl_l)
        ce_vals.append(ce_l)
        
    axes[0].plot(lambda_grid, ce_vals, color='#2563EB', linewidth=2.4, label='Cross-Entropy H(P, Q_λ) [Bits]')
    axes[0].plot(lambda_grid, kl_vals, color='#EF4444', linewidth=2.4, label='KL Divergence D_KL(P || Q_λ) [Bits]')
    axes[0].axhline(h_true, color='#10B981', linestyle='--', linewidth=2.0, label=f'Shannon Entropy Floor H(P) = {h_true:.2f} bits')
    
    min_idx = np.argmin(kl_vals)
    opt_lambda = lambda_grid[min_idx]
    axes[0].plot(opt_lambda, kl_vals[min_idx], marker='o', markersize=8, color='#EF4444')
    axes[0].annotate(f'Minimum KL Divergence\n(λ* = {opt_lambda:.2f})', 
                     xy=(opt_lambda, kl_vals[min_idx]), xytext=(opt_lambda + 1.2, kl_vals[min_idx] + 0.8),
                     arrowprops=dict(facecolor='#1F2937', arrowstyle='->', lw=1.5), fontname='Cambria', fontsize=9.5)
                     
    axes[0].set_title("(A) Convex Geometry of Cross-Entropy & KL Divergence vs. Poisson λ", pad=12)
    axes[0].set_xlabel("Candidate Poisson Mean Parameter (λ)", labelpad=8)
    axes[0].set_ylabel("Information Measure (Bits)", labelpad=8)
    axes[0].set_ylim(0, 8.5)
    
    # Subplot B: Bivariate KL Divergence Contour of Negative Binomial (r, p)
    set_minimal_theme(axes[1])
    r_grid = np.linspace(1.0, 5.0, 50)
    p_grid = np.linspace(0.15, 0.55, 50)
    R_mesh, P_mesh = np.meshgrid(r_grid, p_grid)
    KL_surface = np.zeros_like(R_mesh)
    
    for i in range(len(p_grid)):
        for j in range(len(r_grid)):
            q_nb_pt = stats.nbinom.pmf(k_full, R_mesh[i, j], P_mesh[i, j])
            _, _, kl_pt = compute_kl_divergence_vector(p_true, q_nb_pt)
            KL_surface[i, j] = kl_pt
            
    cp = axes[1].contourf(R_mesh, P_mesh, KL_surface, levels=np.linspace(0, 1.2, 13), cmap='viridis', alpha=0.85)
    cbar = fig.colorbar(cp, ax=axes[1], orientation='vertical', pad=0.03)
    cbar.set_label("KL Divergence D_KL(P || Q_NB) [Bits]", fontname='Cambria', fontsize=9.5)
    
    # Mark optimum
    axes[1].plot(r_nb_tr, p_nb_tr, marker='*', markersize=14, color='#EF4444', markeredgecolor='w', label=f'Optimal NB (r={r_nb_tr:.2f}, p={p_nb_tr:.2f})')
    axes[1].set_title("(B) Information Divergence Landscape D_KL(P || Q_NB) in (r, p) Parameter Space", pad=12)
    axes[1].set_xlabel("Negative Binomial Dispersion Parameter (r)", labelpad=8)
    axes[1].set_ylabel("Negative Binomial Probability Parameter (p)", labelpad=8)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    handles6_a, labels6_a = axes[0].get_legend_handles_labels()
    handles6_b, labels6_b = axes[1].get_legend_handles_labels()
    all_h6 = [handles6_a[0], handles6_a[1], handles6_a[2], handles6_b[0]]
    all_l6 = [labels6_a[0], labels6_a[1], labels6_a[2], labels6_b[0]]
    fig.legend(handles=all_h6, labels=all_l6, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph6_path = os.path.join(module2_dir, '06_information_theory_geometry_entropy_kl.png')
    plt.savefig(graph6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph6_path}")
    
    print("\nModule 2 execution completed successfully!")

if __name__ == '__main__':
    run_module_2()
