"""
Module 1: Data Collection & Preprocessing
Scenario: An image-classification model predicts probabilities for multiple object categories.
Calculate categorical cross-entropy between the true class labels and predicted probabilities.
Analyze Bias-Variance Tradeoff, Overfitting, and Underfitting.

This script:
1. Synthesizes a realistic computer vision dataset (N = 1,000 image samples, C = 5 object classes).
   Classes: Airplane, Automobile, Bird, Cat, Deer.
   Features: 8 deep visual descriptors (Edge Contrast, Texture Energy, Color Saturation,
   Spatial Symmetry, Aspect Ratio Proxy, High Frequency Detail, Dominant Hue, Background Uniformity).
2. Generates ground-truth categorical labels and one-hot target representations.
3. Performs rigorous exploratory data analysis (EDA), data validation, and Z-score standardization.
4. Exports raw, standardized, and class metadata datasets to CSV format.
5. Generates 3 publication-grade visualization figures adhering strictly to formatting standards:
   - Cambria font, minimalist theme with light grid (#E5E7EB), subtle borders (#D1D5DB).
   - Professional hex palette (#2563EB, #10B981, #EF4444, #F59E0B, #8B5CF6).
   - Clearly positioned legends at the bottom.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# Professional color palette
PALETTE = ['#2563EB', '#10B981', '#EF4444', '#F59E0B', '#8B5CF6']
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer']

def set_minimal_theme(ax):
    """Applies a clean, minimal styling with subtle gridlines to an axis."""
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#E5E7EB', zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#D1D5DB')

def synthesize_image_classification_data(n_samples=1000, seed=42):
    """
    Synthesizes N = 1,000 image feature representations across C = 5 distinct object classes.
    Uses class-conditional Gaussian distributions with realistic semantic cross-correlations.
    """
    np.random.seed(seed)
    n_classes = len(CLASS_NAMES)
    samples_per_class = n_samples // n_classes
    
    # Feature names representing extracted deep visual embeddings / morphological descriptors
    feature_names = [
        'Edge_Contrast_Index',
        'Texture_Energy_Score',
        'Color_Saturation_Mean',
        'Spatial_Symmetry_Ratio',
        'Aspect_Ratio_Proxy',
        'High_Freq_Detail',
        'Dominant_Hue_Intensity',
        'Background_Uniformity'
    ]
    
    # Class-conditional prototype means in latent visual feature space
    # 0: Airplane (high contrast, high symmetry, high high-freq detail, uniform sky background)
    # 1: Automobile (high contrast, rectangular aspect ratio, low background uniformity)
    # 2: Bird (organic texture, natural hue, high background variability, compact)
    # 3: Cat (fur texture energy, organic symmetry, warm color saturation, high detail)
    # 4: Deer (high texture energy, tall aspect ratio proxy, earth-tone hues, natural background)
    class_prototypes = np.array([
        [0.85, 0.35, 0.40, 0.88, 1.45, 0.78, 0.65, 0.82],  # Airplane
        [0.90, 0.45, 0.55, 0.82, 1.85, 0.85, 0.50, 0.40],  # Automobile
        [0.45, 0.65, 0.70, 0.55, 0.95, 0.60, 0.75, 0.55],  # Bird
        [0.55, 0.82, 0.65, 0.68, 1.05, 0.72, 0.45, 0.48],  # Cat
        [0.50, 0.78, 0.58, 0.60, 1.35, 0.68, 0.38, 0.62],  # Deer
    ])
    
    feature_stds = np.array([0.14, 0.12, 0.13, 0.11, 0.18, 0.13, 0.12, 0.14])
    
    all_features = []
    all_labels = []
    all_class_names = []
    sample_ids = []
    
    for c_idx, c_name in enumerate(CLASS_NAMES):
        # Generate correlated Gaussian perturbations for each class
        raw_cluster = np.random.normal(
            loc=class_prototypes[c_idx],
            scale=feature_stds,
            size=(samples_per_class, len(feature_names))
        )
        
        # Add realistic non-linear feature interactions and slight noise
        raw_cluster[:, 0] = np.clip(raw_cluster[:, 0] + 0.05 * raw_cluster[:, 4], 0.1, 1.5)
        raw_cluster[:, 1] = np.clip(raw_cluster[:, 1] + 0.04 * raw_cluster[:, 5], 0.05, 1.4)
        raw_cluster[:, 2] = np.clip(raw_cluster[:, 2], 0.05, 1.2)
        raw_cluster[:, 3] = np.clip(raw_cluster[:, 3], 0.2, 1.0)
        raw_cluster[:, 4] = np.clip(raw_cluster[:, 4], 0.5, 2.5)
        raw_cluster[:, 5] = np.clip(raw_cluster[:, 5], 0.1, 1.4)
        raw_cluster[:, 6] = np.clip(raw_cluster[:, 6], 0.05, 1.0)
        raw_cluster[:, 7] = np.clip(raw_cluster[:, 7], 0.1, 1.1)
        
        all_features.append(raw_cluster)
        all_labels.extend([c_idx] * samples_per_class)
        all_class_names.extend([c_name] * samples_per_class)
        sample_ids.extend([f"IMG_{c_idx*samples_per_class + i + 1:04d}" for i in range(samples_per_class)])
        
    X_raw = np.vstack(all_features)
    y_raw = np.array(all_labels)
    
    # Shuffle dataset deterministically while keeping indexing intact
    perm = np.random.permutation(n_samples)
    X_raw = X_raw[perm]
    y_raw = y_raw[perm]
    all_class_names = [all_class_names[i] for i in perm]
    sample_ids = [sample_ids[i] for i in perm]
    
    df_raw = pd.DataFrame(X_raw, columns=feature_names)
    df_raw.insert(0, 'Sample_ID', sample_ids)
    df_raw['Class_Label'] = y_raw
    df_raw['Class_Name'] = all_class_names
    
    # Create one-hot encoded columns for Categorical Cross-Entropy targets
    for c_idx, c_name in enumerate(CLASS_NAMES):
        df_raw[f'Target_{c_name}'] = (df_raw['Class_Label'] == c_idx).astype(int)
        
    return df_raw, feature_names

def run_module_1():
    print("=" * 80)
    print("EXPERIMENT 4 - MODULE 1: DATA COLLECTION & PREPROCESSING")
    print("Multiclass Image Feature Synthesis, CCE Target Encoding & EDA")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Synthesize Dataset
    print("\n[Step 1/5] Synthesizing Multiclass Image Classification Dataset (N=1,000, C=5)...")
    df_raw, feature_names = synthesize_image_classification_data(n_samples=1000, seed=42)
    print(f"Dataset generated: {df_raw.shape[0]} samples across {len(CLASS_NAMES)} classes.")
    print("Class Distribution:")
    for c_idx, c_name in enumerate(CLASS_NAMES):
        cnt = (df_raw['Class_Label'] == c_idx).sum()
        print(f"  - Class {c_idx} ({c_name}): {cnt} samples ({cnt/len(df_raw)*100:.1f}%)")
        
    # 2. Data Validation & Preprocessing
    print("\n[Step 2/5] Performing Data Validation and Z-score Standardization...")
    null_counts = df_raw[feature_names].isnull().sum().sum()
    print(f"Missing Values: {null_counts} (Data integrity confirmed)")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_raw[feature_names])
    
    df_preprocessed = df_raw.copy()
    df_preprocessed[feature_names] = X_scaled
    
    # 3. Create Class Metadata & Summary Table
    print("\n[Step 3/5] Computing Class Feature Profiles and Metadata...")
    summary_list = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        subset = df_raw[df_raw['Class_Label'] == c_idx][feature_names]
        means = subset.mean()
        stds = subset.std()
        summary_list.append({
            'Class_ID': c_idx,
            'Class_Name': c_name,
            'Sample_Count': len(subset),
            'Mean_Edge_Contrast': round(means['Edge_Contrast_Index'], 4),
            'Mean_Texture_Energy': round(means['Texture_Energy_Score'], 4),
            'Mean_Color_Saturation': round(means['Color_Saturation_Mean'], 4),
            'Mean_Spatial_Symmetry': round(means['Spatial_Symmetry_Ratio'], 4),
            'Mean_Aspect_Ratio': round(means['Aspect_Ratio_Proxy'], 4),
            'Mean_High_Freq_Detail': round(means['High_Freq_Detail'], 4),
            'Mean_Hue_Intensity': round(means['Dominant_Hue_Intensity'], 4),
            'Mean_Background_Uniformity': round(means['Background_Uniformity'], 4)
        })
    df_metadata = pd.DataFrame(summary_list)
    
    # 4. Save CSV Datasets
    print("\n[Step 4/5] Exporting CSV Datasets...")
    raw_path = os.path.join(base_dir, 'raw_image_classification_features_1000.csv')
    prep_path = os.path.join(base_dir, 'preprocessed_image_features_1000.csv')
    meta_path = os.path.join(base_dir, 'class_metadata_and_labels_1000.csv')
    
    df_raw.to_csv(raw_path, index=False)
    df_preprocessed.to_csv(prep_path, index=False)
    df_metadata.to_csv(meta_path, index=False)
    print(f"Saved: {raw_path}")
    print(f"Saved: {prep_path}")
    print(f"Saved: {meta_path}")
    
    # 5. Generate Figures
    print("\n[Step 5/5] Generating Module 1 Publication Visualizations...")
    
    # -------------------------------------------------------------
    # Figure 1: Class Distribution & 2D PCA Latent Space Projection
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Class Distribution Bar Chart
    set_minimal_theme(axes[0])
    class_counts = df_raw['Class_Name'].value_counts()[CLASS_NAMES]
    bars = axes[0].bar(CLASS_NAMES, class_counts.values, color=PALETTE, edgecolor='#1F2937', linewidth=0.8, alpha=0.88)
    axes[0].set_title("(A) Balanced Multiclass Image Dataset Distribution (N = 1,000)", pad=12)
    axes[0].set_xlabel("Object Category", labelpad=8)
    axes[0].set_ylabel("Number of Image Samples", labelpad=8)
    axes[0].set_ylim(0, 260)
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 6,
                     f"{height} ({height/len(df_raw)*100:.1f}%)",
                     ha='center', va='bottom', fontname='Cambria', fontsize=10, fontweight='bold')
                     
    # Subplot B: PCA Latent Feature Projection
    set_minimal_theme(axes[1])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100
    
    for c_idx, c_name in enumerate(CLASS_NAMES):
        mask = (df_raw['Class_Label'] == c_idx)
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=PALETTE[c_idx], label=c_name,
                        alpha=0.72, edgecolors='#FFFFFF', linewidths=0.5, s=48)
        # Class Centroid Marker
        centroid = X_pca[mask].mean(axis=0)
        axes[1].scatter(centroid[0], centroid[1], color='#111827', s=120, marker='X', edgecolors='#FFFFFF', linewidths=1.2)
        axes[1].text(centroid[0] + 0.15, centroid[1] + 0.15, f"{c_name} Centroid",
                     fontname='Cambria', fontsize=9.5, fontweight='bold', color='#111827')
                     
    axes[1].set_title(f"(B) 2D PCA Latent Space Feature Embedding (Exp Var: {var_exp[0]:.1f}% + {var_exp[1]:.1f}%)", pad=12)
    axes[1].set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% Variance)", labelpad=8)
    axes[1].set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% Variance)", labelpad=8)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles = [
        mpl.patches.Patch(facecolor=PALETTE[i], edgecolor='#1F2937', label=f"Class {i}: {name}")
        for i, name in enumerate(CLASS_NAMES)
    ]
    legend_handles.append(
        mpl.lines.Line2D([0], [0], marker='X', color='w', markerfacecolor='#111827', markersize=10, label='Class Centroid')
    )
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=6, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig1_path = os.path.join(base_dir, '01_class_distribution_and_feature_embeddings.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Multiclass Feature Separability & Correlation Heatmap
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Violin Plot of Discriminative Features
    set_minimal_theme(axes[0])
    violin_data = []
    violin_labels = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        subset = df_raw[df_raw['Class_Label'] == c_idx]['Edge_Contrast_Index']
        violin_data.append(subset)
        violin_labels.append(c_name)
    parts = axes[0].violinplot(violin_data, showmeans=True, showmedians=False, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE[i])
        pc.set_edgecolor('#1F2937')
        pc.set_alpha(0.75)
    parts['cmeans'].set_color('#111827')
    parts['cmeans'].set_linewidth(1.5)
    parts['cbars'].set_color('#9CA3AF')
    parts['cmaxes'].set_color('#9CA3AF')
    parts['cmins'].set_color('#9CA3AF')
    axes[0].set_xticks(np.arange(1, len(CLASS_NAMES) + 1))
    axes[0].set_xticklabels(CLASS_NAMES)
    axes[0].set_title("(A) Edge Contrast Index Distribution Across Object Classes", pad=12)
    axes[0].set_xlabel("Object Category", labelpad=8)
    axes[0].set_ylabel("Raw Edge Contrast Metric", labelpad=8)
    
    # Subplot B: Pearson Correlation Heatmap
    set_minimal_theme(axes[1])
    corr = df_raw[feature_names].corr()
    short_feat_names = ['Contrast', 'Texture', 'ColorSat', 'Symmetry', 'AspectR', 'HiFreq', 'HueInt', 'BgUniform']
    sns.heatmap(corr, ax=axes[1], annot=True, fmt=".2f", cmap='coolwarm', vmin=-1.0, vmax=1.0, center=0.0,
                xticklabels=short_feat_names, yticklabels=short_feat_names,
                cbar_kws={'orientation': 'horizontal', 'pad': 0.16, 'label': 'Pearson Correlation Coefficient (r)'})
    axes[1].set_title("(B) Visual Feature Cross-Correlation Matrix", pad=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles2 = [
        mpl.patches.Patch(facecolor=PALETTE[i], edgecolor='#1F2937', label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]
    legend_handles2.append(
        mpl.lines.Line2D([0], [0], color='#111827', linewidth=1.5, label='Class Mean Marker')
    )
    fig.legend(handles=legend_handles2, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=6, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig2_path = os.path.join(base_dir, '02_multiclass_feature_separability_and_distributions.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig2_path}")
    
    # -------------------------------------------------------------
    # Figure 3: Feature Standardization & Outlier Distribution
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Pre vs Post Standardization Density Comparison
    set_minimal_theme(axes[0])
    sns.kdeplot(df_raw['Aspect_Ratio_Proxy'], ax=axes[0], color='#EF4444', linewidth=2.2,
                label='Raw Aspect Ratio Proxy (Mean=1.33, Var=0.15)', fill=True, alpha=0.18)
    sns.kdeplot(df_preprocessed['Aspect_Ratio_Proxy'], ax=axes[0], color='#2563EB', linewidth=2.2,
                label='Standardized Aspect Ratio (Mean=0.00, Var=1.00)', fill=True, alpha=0.18)
    sns.kdeplot(df_raw['High_Freq_Detail'], ax=axes[0], color='#F59E0B', linewidth=2.2,
                label='Raw High Frequency Detail (Mean=0.73, Var=0.03)', fill=True, alpha=0.18)
    sns.kdeplot(df_preprocessed['High_Freq_Detail'], ax=axes[0], color='#10B981', linewidth=2.2,
                label='Standardized High Freq Detail (Mean=0.00, Var=1.00)', fill=True, alpha=0.18)
    axes[0].set_title("(A) Probability Density: Raw vs. Standardized Feature Scales", pad=12)
    axes[0].set_xlabel("Feature Value", labelpad=8)
    axes[0].set_ylabel("Kernel Density Estimate", labelpad=8)
    axes[0].set_xlim(-4.0, 5.0)
    
    # Subplot B: Standardized Feature Boxplots
    set_minimal_theme(axes[1])
    box_data = [df_preprocessed[col] for col in feature_names]
    bp = axes[1].boxplot(box_data, patch_artist=True, tick_labels=short_feat_names,
                         medianprops=dict(color='#111827', linewidth=1.5),
                         flierprops=dict(marker='o', markersize=4, markerfacecolor='#EF4444', markeredgecolor='none', alpha=0.6))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('#E0E7FF')
        patch.set_edgecolor('#3730A3')
        patch.set_linewidth(1.0)
    axes[1].axhline(0, color='#111827', linestyle='--', linewidth=0.8, alpha=0.6)
    axes[1].axhline(3, color='#EF4444', linestyle=':', linewidth=0.8, alpha=0.7)
    axes[1].axhline(-3, color='#EF4444', linestyle=':', linewidth=0.8, alpha=0.7)
    axes[1].set_title("(B) Standardized Feature Space Boxplot & Outlier Bounds (Z = +/-3)", pad=12)
    axes[1].set_xlabel("Engineered Deep Feature Descriptors", labelpad=8)
    axes[1].set_ylabel("Standardized Z-Score", labelpad=8)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(-4.5, 4.5)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles3 = [
        mpl.lines.Line2D([0], [0], color='#EF4444', linewidth=2, label='Raw Feature Scale'),
        mpl.lines.Line2D([0], [0], color='#2563EB', linewidth=2, label='Z-Score Standardized Feature (N(0, 1))'),
        mpl.patches.Patch(facecolor='#E0E7FF', edgecolor='#3730A3', label='Interquartile Range (IQR)'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EF4444', markersize=6, label='Tukey Outlier (|Z| > 2.5)')
    ]
    fig.legend(handles=legend_handles3, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig3_path = os.path.join(base_dir, '03_image_feature_standardization_and_outliers.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig3_path}")
    
    print("\nModule 1 processing and visualization completed successfully!")

if __name__ == '__main__':
    run_module_1()
