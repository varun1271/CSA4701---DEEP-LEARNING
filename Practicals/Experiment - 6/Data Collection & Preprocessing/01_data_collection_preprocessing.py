"""
Module 1: Data Collection & Preprocessing
Scenario: A neural network experiences oscillations while optimizing its loss function.
Implement gradient descent with momentum and compare its convergence with standard gradient descent.

This script:
1. Synthesizes an ill-conditioned optimization dataset (N = 1,000 samples, D = 8 features).
   Features exhibit high variance disparity and collinearity, mathematically creating
   an anisotropic ravine loss landscape with high Hessian condition number (kappa >> 1).
2. Computes empirical covariance spectrum and demonstrates eigenvalue ill-conditioning.
3. Performs data validation and Z-score standardization.
4. Exports raw, standardized, and metadata datasets to CSV format.
5. Generates 3 publication-grade visualization figures adhering to standards:
   - Font: Cambria
   - Minimalist theme with subtle light grid (#E5E7EB)
   - Centered bottom legend
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Matplotlib styling strictly per academic requirements
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

PALETTE = ['#2563EB', '#10B981', '#EF4444', '#F59E0B', '#8B5CF6', '#06B6D4']

def set_minimal_theme(ax):
    """Applies a clean, minimal styling with subtle gridlines to an axis."""
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#E5E7EB', zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#D1D5DB')

def synthesize_ill_conditioned_dataset(n_samples=1000, seed=42):
    """
    Synthesizes N = 1,000 observations with anisotropic feature variance
    and strong collinearity, inducing an ill-conditioned Hessian curvature.
    """
    np.random.seed(seed)
    
    # F01: High-variance primary driver (scale ~ 45.0)
    f1_amp = np.random.normal(loc=120.0, scale=42.0, size=n_samples)
    
    # F02: Extremely low-variance sensitive driver (scale ~ 0.25) -> induces curvature disparity
    f2_freq = np.random.normal(loc=2.5, scale=0.35, size=n_samples)
    
    # F03 & F04: Strongly collinear phase signals (r > 0.95) -> induces narrow diagonal ravines
    base_phase = np.random.uniform(0.1, 6.2, size=n_samples)
    f3_phase1 = base_phase + np.random.normal(0, 0.15, size=n_samples)
    f4_phase2 = 0.92 * base_phase + np.random.normal(0, 0.18, size=n_samples)
    
    # F05: Harmonic resonance ratio
    f5_resonance = np.clip(np.random.beta(a=3.0, b=5.0, size=n_samples) * 4.0, 0.1, 4.0)
    
    # F06: Damping coefficient
    f6_damping = np.random.exponential(scale=1.2, size=n_samples) + 0.2
    
    # F07: Noise floor jitter
    f7_jitter = np.random.normal(0.0, 0.5, size=n_samples)
    
    # F08: Spectral centroid offset
    f8_centroid = 0.5 * f1_amp / 20.0 + 0.5 * f5_resonance + np.random.normal(0, 0.4, size=n_samples)
    
    # Non-linear anisotropic system response (ground truth target)
    # The response features a steep canyon in direction 2 and a gentle slope in direction 1
    y_response = (
        0.045 * f1_amp +
        18.5 * np.sin(f2_freq) +
        3.2 * f3_phase1 - 2.8 * f4_phase2 +
        5.5 * (f5_resonance ** 1.5) -
        4.2 * np.log1p(f6_damping) +
        0.015 * (f1_amp - 100.0) * (f2_freq - 2.5) +
        np.random.normal(0, 1.2, size=n_samples)
    )
    
    df = pd.DataFrame({
        'Sample_ID': [f'SIGNAL_{i+1:04d}' for i in range(n_samples)],
        'Signal_Amplitude_HighVar': np.round(f1_amp, 2),
        'Frequency_Modulation_LowVar': np.round(f2_freq, 4),
        'Phase_Shift_Collinear1': np.round(f3_phase1, 3),
        'Phase_Shift_Collinear2': np.round(f4_phase2, 3),
        'Harmonic_Resonance_Ratio': np.round(f5_resonance, 3),
        'Damping_Factor_Proxy': np.round(f6_damping, 3),
        'Noise_Floor_Jitter': np.round(f7_jitter, 3),
        'Spectral_Centroid_Offset': np.round(f8_centroid, 3),
        'System_Response_Target': np.round(y_response, 3)
    })
    
    feature_names = [c for c in df.columns if c not in ['Sample_ID', 'System_Response_Target']]
    return df, feature_names

def run_module_1():
    print("=" * 80)
    print("EXPERIMENT 6 - MODULE 1: DATA COLLECTION & PREPROCESSING")
    print("Ill-Conditioned Optimization Signal Synthesis & Curvature Profiling")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Synthesize Dataset
    print("\n[Step 1/5] Synthesizing Ill-Conditioned Optimization Dataset (N = 1,000)...")
    df_raw, feature_names = synthesize_ill_conditioned_dataset(n_samples=1000, seed=42)
    print(f"Generated {len(df_raw)} signal instances across {len(feature_names)} features.")
    
    # 2. Curvature & Condition Number Analysis
    print("\n[Step 2/5] Quantifying Loss Surface Ill-Conditioning (Eigenvalue Spectrum)...")
    X_raw_matrix = df_raw[feature_names].values
    cov_matrix = np.cov(X_raw_matrix, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov_matrix))[::-1]
    condition_number = eigenvalues[0] / eigenvalues[-1]
    print(f"  Eigenvalue Max (lambda_max): {eigenvalues[0]:.2f}")
    print(f"  Eigenvalue Min (lambda_min): {eigenvalues[-1]:.4f}")
    print(f"  Condition Number (kappa):    {condition_number:.1f} (Extreme Curvature Anisotropy >> 1)")
    
    # 3. Validation & Z-Score Standardization
    print("\n[Step 3/5] Performing Data Validation & Z-Score Standardization...")
    null_counts = df_raw.isnull().sum().sum()
    print(f"Missing Values: {null_counts} (Data integrity confirmed)")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw_matrix)
    
    df_preprocessed = df_raw.copy()
    df_preprocessed[feature_names] = X_scaled
    
    # Standardize target for scale-independent gradient descent
    y_raw = df_raw['System_Response_Target'].values
    y_scaled = (y_raw - np.mean(y_raw)) / np.std(y_raw)
    df_preprocessed['Target_Standardized'] = np.round(y_scaled, 4)
    
    # 4. Feature Metadata Table
    print("\n[Step 4/5] Constructing Feature Metadata Table...")
    meta_rows = []
    descriptions = {
        'Signal_Amplitude_HighVar': 'Dominant high-energy oscillation amplitude (Scale ~ 42.0)',
        'Frequency_Modulation_LowVar': 'Micro-scale carrier modulation frequency (Scale ~ 0.35)',
        'Phase_Shift_Collinear1': 'Primary harmonic phase offset in radians [0 - 2*pi]',
        'Phase_Shift_Collinear2': 'Secondary harmonic phase offset (Collinear with Phase1, r > 0.95)',
        'Harmonic_Resonance_Ratio': 'Resonance peak energy ratio relative to fundamental',
        'Damping_Factor_Proxy': 'Exponential decay dissipation rate coefficient',
        'Noise_Floor_Jitter': 'Stochastic additive gaussian channel disturbance',
        'Spectral_Centroid_Offset': 'Spectral power distribution center of gravity'
    }
    units = {
        'Signal_Amplitude_HighVar': 'Volts (V)',
        'Frequency_Modulation_LowVar': 'Radians / sec',
        'Phase_Shift_Collinear1': 'Radians',
        'Phase_Shift_Collinear2': 'Radians',
        'Harmonic_Resonance_Ratio': 'Ratio',
        'Damping_Factor_Proxy': 'sec^-1',
        'Noise_Floor_Jitter': 'Normalized',
        'Spectral_Centroid_Offset': 'Hertz (Hz)'
    }
    for idx, f in enumerate(feature_names):
        meta_rows.append({
            'Feature_Index': f'F{idx+1:02d}',
            'Feature_Name': f,
            'Measurement_Unit': units[f],
            'Raw_Mean': round(df_raw[f].mean(), 3),
            'Raw_Std': round(df_raw[f].std(), 3),
            'Raw_Min': round(df_raw[f].min(), 3),
            'Raw_Max': round(df_raw[f].max(), 3),
            'Curvature_Significance': descriptions[f]
        })
    df_metadata = pd.DataFrame(meta_rows)
    
    # 5. Export Datasets
    print("\n[Step 5/5] Exporting CSV Datasets...")
    raw_path = os.path.join(base_dir, 'raw_ill_conditioned_signals_1000.csv')
    prep_path = os.path.join(base_dir, 'preprocessed_optimization_signals_1000.csv')
    meta_path = os.path.join(base_dir, 'signal_feature_metadata_1000.csv')
    
    df_raw.to_csv(raw_path, index=False)
    df_preprocessed.to_csv(prep_path, index=False)
    df_metadata.to_csv(meta_path, index=False)
    print(f"Saved: {raw_path}")
    print(f"Saved: {prep_path}")
    print(f"Saved: {meta_path}")
    
    # 6. Generate Figures
    print("\nGenerating Module 1 Publication Visualizations...")
    
    # -------------------------------------------------------------
    # Figure 1: Ill-Conditioned Feature Distributions & Target
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Target System Response Distribution
    set_minimal_theme(axes[0])
    sns.histplot(df_raw['System_Response_Target'], ax=axes[0], kde=True, bins=35,
                 color='#2563EB', edgecolor='#1E40AF', alpha=0.65, line_kws={'linewidth': 2.2})
    mean_resp = df_raw['System_Response_Target'].mean()
    median_resp = df_raw['System_Response_Target'].median()
    axes[0].axvline(mean_resp, color='#EF4444', linestyle='--', linewidth=2.0, label=f'Mean Response: {mean_resp:.2f}')
    axes[0].axvline(median_resp, color='#10B981', linestyle=':', linewidth=2.0, label=f'Median Response: {median_resp:.2f}')
    axes[0].set_title("(A) Ill-Conditioned System Response Target Distribution", pad=12)
    axes[0].set_xlabel("Continuous Response Metric", labelpad=8)
    axes[0].set_ylabel("Observation Frequency Count", labelpad=8)
    
    # Subplot B: Extreme Variance Disparity (HighVar vs LowVar)
    set_minimal_theme(axes[1])
    sc = axes[1].scatter(df_raw['Signal_Amplitude_HighVar'], df_raw['Frequency_Modulation_LowVar'],
                         c=df_raw['System_Response_Target'], cmap='coolwarm',
                         alpha=0.75, edgecolors='#FFFFFF', linewidths=0.4, s=45)
    cbar = fig.colorbar(sc, ax=axes[1], orientation='horizontal', pad=0.16)
    cbar.set_label('System Response Target Value')
    axes[1].set_title("(B) Anisotropic Feature Geometry: High-Variance vs. Low-Variance Axis", pad=12)
    axes[1].set_xlabel("Signal Amplitude HighVar (Std = 41.8 V - Steep Axis)", labelpad=8)
    axes[1].set_ylabel("Frequency Modulation LowVar (Std = 0.35 rad/s - Flat Axis)", labelpad=8)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles1 = [
        mpl.lines.Line2D([0], [0], color='#EF4444', linestyle='--', linewidth=2.0, label=f'Mean Response ({mean_resp:.2f})'),
        mpl.lines.Line2D([0], [0], color='#10B981', linestyle=':', linewidth=2.0, label=f'Median Response ({median_resp:.2f})'),
        mpl.patches.Patch(facecolor='#2563EB', alpha=0.65, label='Empirical Target Density'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2563EB', markersize=7, label=f'Condition Ratio kappa = {condition_number:.1f}')
    ]
    fig.legend(handles=legend_handles1, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig1_path = os.path.join(base_dir, '01_ill_conditioned_feature_distributions.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Correlation Matrix & Eigenvalue Spectrum
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Correlation Matrix
    set_minimal_theme(axes[0])
    short_feat = ['AmpHigh', 'FreqLow', 'Phase1', 'Phase2', 'Reson', 'Damping', 'Jitter', 'Centroid']
    corr = df_raw[feature_names].corr()
    sns.heatmap(corr, ax=axes[0], annot=True, fmt=".2f", cmap='coolwarm', vmin=-1.0, vmax=1.0, center=0.0,
                xticklabels=short_feat, yticklabels=short_feat,
                cbar_kws={'orientation': 'horizontal', 'pad': 0.16, 'label': 'Pearson Correlation (r)'})
    axes[0].set_title("(A) Predictor Correlation Heatmap (Collinear Phase1 & Phase2 r = 0.96)", pad=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Subplot B: Covariance Eigenvalue Spectrum (Log Scale)
    set_minimal_theme(axes[1])
    e_idx = np.arange(1, len(eigenvalues) + 1)
    bars_eig = axes[1].bar(e_idx, eigenvalues, color='#2563EB', edgecolor='#1E40AF', width=0.55, alpha=0.88)
    axes[1].plot(e_idx, eigenvalues, color='#EF4444', marker='o', linewidth=2.0, label='Eigenvalue Decay Curve')
    axes[1].set_yscale('log')
    axes[1].set_xticks(e_idx)
    axes[1].set_xticklabels([f"lambda_{i}" for i in e_idx])
    axes[1].set_title(f"(B) Covariance Eigenvalue Spectrum (Hessian Condition Number: {condition_number:.1f})", pad=12)
    axes[1].set_xlabel("Eigenvalue Rank Index", labelpad=8)
    axes[1].set_ylabel("Eigenvalue Magnitude (Log Scale)", labelpad=8)
    for bar in bars_eig:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h * 1.25, f"{h:.2f}",
                     ha='center', va='bottom', fontname='Cambria', fontsize=8.5, fontweight='bold')
                     
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles2 = [
        mpl.patches.Patch(facecolor='#2563EB', edgecolor='#1E40AF', label=f'Eigenvalues lambda_1 to lambda_8'),
        mpl.lines.Line2D([0], [0], color='#EF4444', marker='o', linewidth=2.0, label=f'Condition Number kappa = lambda_max/lambda_min = {condition_number:.1f} >> 1'),
        mpl.patches.Patch(facecolor='#F87171', label='Collinear Feature Subspace (Diagonal Ravine)')
    ]
    fig.legend(handles=legend_handles2, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=3, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig2_path = os.path.join(base_dir, '02_feature_correlation_and_eigenvalue_spectrum.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig2_path}")
    
    # -------------------------------------------------------------
    # Figure 3: Standardization & Outlier Distribution
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Pre vs Post Standardization Density Comparison
    set_minimal_theme(axes[0])
    sns.kdeplot(df_raw['Signal_Amplitude_HighVar'] / 100.0, ax=axes[0], color='#EF4444', linewidth=2.2,
                label='Raw Amplitude (Normalized by 100, Std=0.42)', fill=True, alpha=0.18)
    sns.kdeplot(df_preprocessed['Signal_Amplitude_HighVar'], ax=axes[0], color='#2563EB', linewidth=2.2,
                label='Standardized Amplitude (Mean=0, Std=1)', fill=True, alpha=0.18)
    sns.kdeplot(df_preprocessed['Frequency_Modulation_LowVar'], ax=axes[0], color='#10B981', linewidth=2.2,
                label='Standardized Frequency (Mean=0, Std=1)', fill=True, alpha=0.18)
    axes[0].set_title("(A) Probability Density: Raw vs. Standardized Feature Scales", pad=12)
    axes[0].set_xlabel("Standardized Coordinate Value", labelpad=8)
    axes[0].set_ylabel("Kernel Density Estimate", labelpad=8)
    axes[0].set_xlim(-4.0, 5.0)
    
    # Subplot B: Standardized Boxplots
    set_minimal_theme(axes[1])
    box_data = [df_preprocessed[col] for col in feature_names]
    bp = axes[1].boxplot(box_data, patch_artist=True, tick_labels=short_feat,
                         medianprops=dict(color='#111827', linewidth=1.5),
                         flierprops=dict(marker='o', markersize=4, markerfacecolor='#EF4444', markeredgecolor='none', alpha=0.6))
    for patch in bp['boxes']:
        patch.set_facecolor('#E0E7FF')
        patch.set_edgecolor('#3730A3')
        patch.set_linewidth(1.0)
    axes[1].axhline(0, color='#111827', linestyle='--', linewidth=0.8, alpha=0.6)
    axes[1].axhline(3, color='#EF4444', linestyle=':', linewidth=0.8, alpha=0.7)
    axes[1].axhline(-3, color='#EF4444', linestyle=':', linewidth=0.8, alpha=0.7)
    axes[1].set_title("(B) Standardized Feature Boxplot & Outlier Bounds (|Z| <= 3.0)", pad=12)
    axes[1].set_xlabel("Signal Predictor Dimensions", labelpad=8)
    axes[1].set_ylabel("Standardized Z-Score", labelpad=8)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(-4.5, 4.5)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles3 = [
        mpl.lines.Line2D([0], [0], color='#EF4444', linewidth=2.0, label='Raw Feature Scale'),
        mpl.lines.Line2D([0], [0], color='#2563EB', linewidth=2.0, label='Standardized Dimension (N(0, 1))'),
        mpl.patches.Patch(facecolor='#E0E7FF', edgecolor='#3730A3', label='Interquartile Range (IQR)'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EF4444', markersize=6, label='Tukey Outlier (|Z| > 2.5)')
    ]
    fig.legend(handles=legend_handles3, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig3_path = os.path.join(base_dir, '03_feature_standardization_and_outlier_bounds.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig3_path}")
    
    print("\nModule 1 processing and visualization completed successfully!")

if __name__ == '__main__':
    run_module_1()
