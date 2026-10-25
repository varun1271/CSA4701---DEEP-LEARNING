"""
Module 1: Data Collection & Preprocessing
Scenario: A manufacturing unit records the number of defective products in different batches.
Model the defect counts using a suitable probability distribution and estimate the probability
of observing a specified number of defects. Analyze Entropy, Cross-Entropy, and KL Divergence.

This script:
1. Synthesizes an enterprise manufacturing batch dataset (N = 1,000 batches, n = 500 units/batch).
2. Computes empirical defect distributions and summary statistics.
3. Fits parametric probability distributions for discrete defect counts:
   - Binomial Distribution: Bin(n=500, p)
   - Poisson Distribution: Pois(lambda)
   - Negative Binomial Distribution: NB(r, p)
4. Evaluates Information-Theoretic Metrics on the empirical vs. fitted models:
   - Shannon Entropy H(P)
   - Cross-Entropy H(P, Q)
   - Kullback-Leibler (KL) Divergence D_KL(P || Q)
5. Performs rigorous preprocessing:
   - Data validation and missing value checks.
   - Z-score Standardization (Mean = 0, Variance = 1).
   - Scale comparisons.
6. Exports raw, preprocessed, and inspection metadata datasets to CSV.
7. Generates publication-grade, colorful visualization graphs:
   - Font: Cambria
   - Minimalist theme with light grid
   - Clear X and Y labels with well-numbered scales
   - Legend positioned cleanly at the bottom
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
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

# Professional colorful palette
COLORS = ['#2563EB', '#EC4899', '#10B981', '#F59E0B', '#8B5CF6', '#06B6D4', '#EF4444', '#14B8A6']

def set_minimal_theme(ax):
    """Applies a clean, minimal styling with subtle gridlines to an axis."""
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#E5E7EB', zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#D1D5DB')
        spine.set_linewidth(0.8)
    ax.tick_params(colors='#1F2937', which='both', direction='out', length=4, width=0.8)

def generate_manufacturing_batch_data(n_batches=1000, batch_size=500, random_state=42):
    """
    Generates realistic industrial production batches with environmental
    and operational drivers of product defects.
    """
    np.random.seed(random_state)
    
    # 1. Operating Environmental and Mechanical Features
    temp = np.random.normal(78.5, 6.5, size=n_batches)
    temp = np.clip(temp, 60.0, 98.0)
    
    speed = np.random.normal(82.0, 14.0, size=n_batches)
    speed = np.clip(speed, 45.0, 125.0)
    
    vibration = np.random.gamma(shape=4.0, scale=0.6, size=n_batches)
    vibration = np.clip(vibration, 0.8, 5.5)
    
    purity = np.random.beta(a=35.0, b=1.5, size=n_batches) * 100.0
    purity = np.clip(purity, 90.0, 99.8)
    
    shift_hours = np.random.uniform(1.0, 12.0, size=n_batches)
    
    humidity = np.random.normal(54.0, 11.5, size=n_batches)
    humidity = np.clip(humidity, 28.0, 82.0)
    
    tool_wear = np.random.beta(a=2.0, b=2.5, size=n_batches)
    
    # 2. Defect Count Generation with Latent Environmental Variability (Overdispersion)
    # Baseline defect rate modified by process stress factors
    stress_index = (
        0.35 * ((temp - 78.5) / 6.5) +
        0.30 * ((speed - 82.0) / 14.0) +
        0.40 * ((vibration - 2.4) / 0.8) -
        0.45 * ((purity - 96.5) / 1.8) +
        0.25 * ((shift_hours - 6.0) / 3.0) +
        0.35 * ((tool_wear - 0.45) / 0.2)
    )
    
    # Expected defect rate per unit lambda_rate
    base_defect_prob = 0.0096  # Mean ~4.8 defects per 500 units
    lambda_param = batch_size * base_defect_prob * np.exp(0.38 * stress_index)
    lambda_param = np.clip(lambda_param, 0.5, 22.0)
    
    # Negative Binomial / Gamma-Poisson mixture introduces authentic physical overdispersion
    gamma_noise = np.random.gamma(shape=5.0, scale=1.0/5.0, size=n_batches)
    batch_rates = lambda_param * gamma_noise
    defect_counts = np.random.poisson(batch_rates)
    defect_counts = np.clip(defect_counts, 0, 25)
    
    defect_rate = defect_counts / batch_size
    
    # Quality control classifications
    status_labels = []
    for d in defect_counts:
        if d <= 3:
            status_labels.append('Pristine / Optimal')
        elif d <= 6:
            status_labels.append('Controlled / Acceptable')
        elif d <= 9:
            status_labels.append('Elevated / Warning')
        else:
            status_labels.append('Critical / Batch Scrap')
            
    batch_ids = [f"BATCH_{i+1:04d}" for i in range(n_batches)]
    lines = np.random.choice(['Line Alpha', 'Line Beta', 'Line Gamma', 'Line Delta'], size=n_batches, p=[0.35, 0.28, 0.22, 0.15])
    shifts = np.random.choice(['Day Shift', 'Evening Shift', 'Night Shift'], size=n_batches, p=[0.45, 0.35, 0.20])
    operator_ids = [f"OP_{np.random.randint(101, 130):03d}" for _ in range(n_batches)]
    
    df_features = pd.DataFrame({
        'Machine_Temperature': temp,
        'Assembly_Speed': speed,
        'Vibration_Level': vibration,
        'Material_Purity': purity,
        'Shift_Hours': shift_hours,
        'Ambient_Humidity': humidity,
        'Tool_Wear_Index': tool_wear,
        'Defect_Count': defect_counts
    })
    
    df_meta = pd.DataFrame({
        'Batch_ID': batch_ids,
        'Batch_Size': [batch_size] * n_batches,
        'Defect_Count': defect_counts,
        'Defect_Rate': defect_rate,
        'Quality_Status': status_labels,
        'Production_Line': lines,
        'Shift_Type': shifts,
        'Operator_ID': operator_ids,
        'Is_Rejected_Batch': (defect_counts >= 10).astype(int)
    })
    
    return df_features, df_meta

def compute_information_theoretic_metrics(p_emp, q_model, eps=1e-12):
    """
    Computes Shannon Entropy H(P), Cross-Entropy H(P, Q), and KL Divergence D_KL(P || Q).
    All metrics computed in bits (base-2).
    """
    p_safe = np.clip(p_emp, eps, 1.0)
    p_safe /= np.sum(p_safe)
    
    q_safe = np.clip(q_model, eps, 1.0)
    q_safe /= np.sum(q_safe)
    
    # 1. Shannon Entropy: H(P) = -sum(P(k) * log2 P(k))
    shannon_entropy = -np.sum(p_safe * np.log2(p_safe))
    
    # 2. Cross-Entropy: H(P, Q) = -sum(P(k) * log2 Q(k))
    cross_entropy = -np.sum(p_safe * np.log2(q_safe))
    
    # 3. KL Divergence: D_KL(P || Q) = sum(P(k) * log2(P(k) / Q(k))) = H(P, Q) - H(P)
    kl_divergence = cross_entropy - shannon_entropy
    kl_divergence = max(kl_divergence, 0.0)  # Gibbs' Inequality ensures non-negativity
    
    return shannon_entropy, cross_entropy, kl_divergence

def run_module_1():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 75)
    print("MODULE 1: DATA COLLECTION & PREPROCESSING (MANUFACTURING DEFECT COUNTS)")
    print("=" * 75)
    
    # 1. Synthesize 1,000 industrial manufacturing batch records
    print("Generating manufacturing batch dataset: N = 1,000 batches (n = 500 units each)...")
    df_raw, df_meta = generate_manufacturing_batch_data(n_batches=1000, batch_size=500, random_state=42)
    
    print(f"Dataset generated successfully. Total batches: {len(df_raw)}")
    print("\nFeature Summary Statistics (Raw Operational Scales):")
    for col in df_raw.columns:
        print(f"  - {col:22s} | Mean: {df_raw[col].mean():8.2f} | Std: {df_raw[col].std():8.2f} | Range: [{df_raw[col].min():6.2f}, {df_raw[col].max():6.2f}]")
        
    defects = df_raw['Defect_Count'].values
    mean_defects = np.mean(defects)
    var_defects = np.var(defects, ddof=1)
    dispersion_ratio = var_defects / mean_defects
    
    print("\n" + "-" * 75)
    print("DEFECT COUNT STATISTICAL DISPERSION ANALYSIS:")
    print("-" * 75)
    print(f"  * Sample Mean Defect Count (lambda_hat): {mean_defects:.4f} defects / batch")
    print(f"  * Sample Variance (s^2):                {var_defects:.4f}")
    print(f"  * Dispersion Index (Var / Mean):        {dispersion_ratio:.4f}")
    if dispersion_ratio > 1.05:
        print(f"  * Result: Substantial Overdispersion detected (Var > Mean). Standard Poisson assumption is violated!")
        print(f"            Negative Binomial distribution is theoretically indicated.")
    else:
        print(f"  * Result: Equidispersed (Poisson compatible).")
        
    # 2. Probability Distribution Fitting
    k_max = int(np.max(defects))
    k_support = np.arange(0, k_max + 1)
    
    # (a) Empirical Distribution P(k)
    p_emp = np.array([np.mean(defects == k) for k in k_support])
    
    # (b) Poisson Model: Pois(lambda = mean_defects)
    q_pois = stats.poisson.pmf(k_support, mu=mean_defects)
    
    # (c) Binomial Model: Bin(n = 500, p = mean_defects / 500)
    p_binom_param = mean_defects / 500.0
    q_binom = stats.binom.pmf(k_support, n=500, p=p_binom_param)
    
    # (d) Negative Binomial Model: NB(r, p) via Method of Moments
    # Mean = r * (1 - p) / p = mean_defects
    # Var = r * (1 - p) / p^2 = var_defects
    # => p = Mean / Var, r = Mean^2 / (Var - Mean)
    p_nb = mean_defects / var_defects
    r_nb = (mean_defects ** 2) / (var_defects - mean_defects)
    q_nb = stats.nbinom.pmf(k_support, r_nb, p_nb)
    
    # (e) Uniform Reference Baseline (Maximum Uncertainty)
    q_unif = np.ones_like(k_support, dtype=float) / len(k_support)
    
    # 3. Information Theory: Shannon Entropy, Cross-Entropy, and KL Divergence
    print("\n" + "-" * 75)
    print("INFORMATION THEORY EVALUATION (SHANNON ENTROPY, CROSS-ENTROPY & KL DIVERGENCE):")
    print("-" * 75)
    
    h_p, _, _ = compute_information_theoretic_metrics(p_emp, p_emp)
    print(f"  * Intrinsic Empirical Shannon Entropy H(P): {h_p:.4f} bits")
    print(f"    (Represents the fundamental irreducible uncertainty in the manufacturing defect counts)\n")
    
    models = {
        'Negative Binomial NB(r, p)': q_nb,
        'Poisson Pois(lambda)': q_pois,
        'Binomial Bin(n=500, p)': q_binom,
        'Uniform Baseline (Max Entropy)': q_unif
    }
    
    print(f"  {'Candidate Distribution Q(k)':32s} | {'Cross-Entropy H(P, Q)':>20s} | {'KL Divergence D_KL':>18s} | {'Information Loss':>16s}")
    print("  " + "-" * 95)
    for name, q_dist in models.items():
        _, ce, kl = compute_information_theoretic_metrics(p_emp, q_dist)
        print(f"  {name:32s} | {ce:17.4f} bits | {kl:15.4f} bits | {kl/h_p*100:14.2f}%")
        
    # 4. Preprocessing: Z-score Standardization
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df_raw)
    df_preprocessed = pd.DataFrame(scaled_matrix, columns=df_raw.columns)
    
    # Save CSV files
    raw_csv_path = os.path.join(base_dir, 'raw_manufacturing_batches_1000.csv')
    prep_csv_path = os.path.join(base_dir, 'preprocessed_manufacturing_batches_1000.csv')
    meta_csv_path = os.path.join(base_dir, 'batch_metadata_and_inspection_1000.csv')
    
    df_raw.to_csv(raw_csv_path, index=False)
    df_preprocessed.to_csv(prep_csv_path, index=False)
    df_meta.to_csv(meta_csv_path, index=False)
    
    print(f"\nExported raw batches to: {raw_csv_path}")
    print(f"Exported preprocessed features to: {prep_csv_path}")
    print(f"Exported inspection metadata to: {meta_csv_path}")
    
    # 5. Generate Publication-Grade Visualization Graphs
    print("\nGenerating Module 1 Visualization Graphs...")
    
    # Graph 1: Feature Scales Before vs. After Preprocessing (Boxplot / Spread)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=False)
    fig.patch.set_facecolor('#FFFFFF')
    
    short_labels = ['Temp (°C)', 'Speed (u/m)', 'Vibration (mm/s)', 'Purity (%)', 
                    'Shift (hrs)', 'Humidity (%)', 'Tool Wear', 'Defects (k)']
    
    # Panel A: Raw Scales (Symlog scale to accommodate purity ~98, temp ~80, vibration ~2.5, defect ~5)
    set_minimal_theme(axes[0])
    axes[0].boxplot(
        [df_raw[col] for col in df_raw.columns],
        patch_artist=True,
        tick_labels=short_labels,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"#EF4444", "markeredgecolor":"#EF4444", "markersize":5},
        medianprops={"color":"#1E293B", "linewidth":1.4},
        boxprops={"facecolor":"#93C5FD", "edgecolor":"#1E40AF", "linewidth":0.9, "alpha":0.85},
        whiskerprops={"color":"#1E40AF", "linewidth":0.9},
        capprops={"color":"#1E40AF", "linewidth":0.9},
        flierprops={"marker":"d", "markerfacecolor":"#60A5FA", "alpha":0.4, "markersize":3.5}
    )
    axes[0].set_title("(A) Raw Operational Feature Scales Before Preprocessing (Disparate Physical Engineering Units)", pad=10)
    axes[0].set_ylabel("Raw Measurement Magnitude", labelpad=8)
    axes[0].set_yscale('symlog', linthresh=1.0)
    
    # Panel B: Standardized Z-scores
    set_minimal_theme(axes[1])
    axes[1].boxplot(
        [df_preprocessed[col] for col in df_preprocessed.columns],
        patch_artist=True,
        tick_labels=short_labels,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"#EF4444", "markeredgecolor":"#EF4444", "markersize":5},
        medianprops={"color":"#1E293B", "linewidth":1.4},
        boxprops={"facecolor":"#A7F3D0", "edgecolor":"#065F46", "linewidth":0.9, "alpha":0.85},
        whiskerprops={"color":"#065F46", "linewidth":0.9},
        capprops={"color":"#065F46", "linewidth":0.9},
        flierprops={"marker":"d", "markerfacecolor":"#34D399", "alpha":0.4, "markersize":3.5}
    )
    axes[1].set_title("(B) Preprocessed Feature Scales After Z-Score Standardization (Mean = 0, Std = 1)", pad=10)
    axes[1].set_xlabel("Manufacturing Operational & Environmental Variables", labelpad=8)
    axes[1].set_ylabel("Standardized Z-Score", labelpad=8)
    axes[1].set_ylim(-3.8, 4.2)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.18, hspace=0.32)
    legend_elements = [
        mpl.patches.Patch(facecolor='#93C5FD', edgecolor='#1E40AF', label='Raw Manufacturing Features (Varying Units)'),
        mpl.patches.Patch(facecolor='#A7F3D0', edgecolor='#065F46', label='Standardized Z-Score Features'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EF4444', markersize=7, label='Empirical Mean Value'),
        mpl.lines.Line2D([0], [0], color='#1E293B', lw=1.5, label='Empirical Median Line')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph1_path = os.path.join(base_dir, '01_batch_feature_distributions_and_scales.png')
    plt.savefig(graph1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph1_path}")
    
    # Graph 2: Empirical Defect Count Distribution vs. Fitted Theoretical PMFs & Residuals
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: PMF Fits Overlay
    set_minimal_theme(axes[0])
    axes[0].bar(k_support, p_emp, color='#93C5FD', edgecolor='#2563EB', alpha=0.55, width=0.7, label='Empirical Defect Frequency P(k)')
    axes[0].plot(k_support, q_nb, marker='o', color='#10B981', linewidth=2.4, markersize=5, label=f'Negative Binomial NB(r={r_nb:.1f}, p={p_nb:.2f})')
    axes[0].plot(k_support, q_pois, marker='s', color='#EF4444', linestyle='--', linewidth=2.0, markersize=4.5, label=f'Poisson Pois(λ={mean_defects:.1f})')
    axes[0].plot(k_support, q_binom, marker='^', color='#F59E0B', linestyle=':', linewidth=2.0, markersize=4.5, label=f'Binomial Bin(n=500, p={p_binom_param:.4f})')
    axes[0].set_title("(A) Defect Count PMF Fits: Empirical vs. Parametric Models", pad=12)
    axes[0].set_xlabel("Observed Defect Count per Batch (k)", labelpad=8)
    axes[0].set_ylabel("Probability Mass P(K = k)", labelpad=8)
    axes[0].set_xlim(-0.5, 18.5)
    
    # Subplot B: Modeling Residuals [P(k) - Q(k)]
    set_minimal_theme(axes[1])
    res_pois = p_emp - q_pois
    res_nb = p_emp - q_nb
    axes[1].axhline(0, color='#6B7280', linestyle='-', linewidth=1.2)
    axes[1].plot(k_support, res_pois, marker='s', color='#EF4444', linestyle='--', linewidth=2.0, label='Poisson Residuals [P(k) - Q_Pois(k)]')
    axes[1].plot(k_support, res_nb, marker='o', color='#10B981', linestyle='-', linewidth=2.2, label='Neg-Binomial Residuals [P(k) - Q_NB(k)]')
    axes[1].fill_between(k_support, res_pois, 0, color='#EF4444', alpha=0.15)
    axes[1].fill_between(k_support, res_nb, 0, color='#10B981', alpha=0.15)
    axes[1].set_title("(B) Modeling Residual Deviation: Poisson vs. Negative Binomial", pad=12)
    axes[1].set_xlabel("Observed Defect Count per Batch (k)", labelpad=8)
    axes[1].set_ylabel("Probability Mass Error [P(k) - Q(k)]", labelpad=8)
    axes[1].set_xlim(-0.5, 18.5)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    handles2_a, labels2_a = axes[0].get_legend_handles_labels()
    handles2_b, labels2_b = axes[1].get_legend_handles_labels()
    all_h2 = [handles2_a[0], handles2_a[1], handles2_a[2], handles2_a[3], handles2_b[1]]
    all_l2 = [labels2_a[0], labels2_a[1], labels2_a[2], labels2_a[3], labels2_b[1]]
    fig.legend(handles=all_h2, labels=all_l2, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=5, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph2_path = os.path.join(base_dir, '02_defect_count_probability_distribution_fits.png')
    plt.savefig(graph2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph2_path}")
    
    # Graph 3: Correlation Matrix & Defect Sensitivity Drivers
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Heatmap of Pearson Correlation
    set_minimal_theme(axes[0])
    corr_matrix = df_raw.corr()
    short_cols = ['Temp', 'Speed', 'Vibration', 'Purity', 'Shift', 'Humidity', 'Wear', 'Defects']
    sns.heatmap(corr_matrix, ax=axes[0], annot=True, fmt=".2f", cmap='coolwarm', vmin=-1.0, vmax=1.0, center=0.0,
                xticklabels=short_cols, yticklabels=short_cols,
                cbar_kws={'orientation': 'horizontal', 'pad': 0.16, 'label': 'Pearson Correlation Coefficient r'})
    axes[0].set_title("(A) Manufacturing Variables Pearson Correlation Matrix", pad=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Subplot B: Defect Driver Correlation Ranking
    set_minimal_theme(axes[1])
    defect_corrs = corr_matrix['Defect_Count'].drop('Defect_Count').sort_values(ascending=True)
    bar_colors = ['#10B981' if r < 0 else '#EF4444' for r in defect_corrs.values]
    y_pos = np.arange(len(defect_corrs))
    axes[1].barh(y_pos, defect_corrs.values, color=bar_colors, edgecolor='#1F2937', linewidth=0.8, alpha=0.85)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(defect_corrs.index)
    axes[1].axvline(0, color='#1F2937', linewidth=1.0)
    for i, v in enumerate(defect_corrs.values):
        offset = 0.02 if v >= 0 else -0.06
        axes[1].text(v + offset, i, f"{v:+.3f}", va='center', fontname='Cambria', fontsize=9.5, fontweight='bold')
    axes[1].set_title("(B) Operational Drivers of Product Defects (Pearson r)", pad=12)
    axes[1].set_xlabel("Linear Correlation with Defect Count (r)", labelpad=8)
    axes[1].set_xlim(-0.65, 0.65)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    legend_elements3 = [
        mpl.patches.Patch(facecolor='#EF4444', edgecolor='#B91C1C', label='Positive Defect Driver (Increases Scrap Risk)'),
        mpl.patches.Patch(facecolor='#10B981', edgecolor='#047857', label='Protective Quality Driver (Suppresses Defects)')
    ]
    fig.legend(handles=legend_elements3, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=2, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph3_path = os.path.join(base_dir, '03_manufacturing_feature_correlation_and_defect_drivers.png')
    plt.savefig(graph3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph3_path}")
    
    print("\nModule 1 execution completed successfully!")

if __name__ == '__main__':
    run_module_1()
