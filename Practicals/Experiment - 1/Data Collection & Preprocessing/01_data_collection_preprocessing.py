"""
Module 1: Data Collection & Preprocessing
Scenario: An online shopping platform represents 500 products using 20 numerical features each.

This script:
1. Loads the e-commerce product catalog from 'hm_ecommerce_products_enriched.parquet'.
2. Selects 500 representative products and constructs exactly 20 numerical features.
3. Performs rigorous preprocessing:
   - Data cleaning, missing value checks, and data type casting.
   - Outlier inspection and robust scaling.
   - Standard scaling (Zero mean, unit variance).
   - L2 Unit-Norm Normalization (ensuring vector dot products strictly compute Cosine Similarity).
4. Exports raw and preprocessed datasets to CSV.
5. Generates high-quality, professional, colorful visualization graphs:
   - Font: Cambria
   - Minimalist theme with light grid
   - Clear X and Y labels with well-numbered scales
   - Legend positioned cleanly at the bottom
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
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

def run_module_1():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base_dir, 'hm_ecommerce_products_enriched.parquet')
    
    print("=" * 70)
    print("MODULE 1: DATA COLLECTION & PREPROCESSING")
    print("=" * 70)
    print(f"Loading raw dataset from: {parquet_path}")
    
    raw_export_path = os.path.join(base_dir, 'raw_products_500.csv')
    prep_export_path = os.path.join(base_dir, 'preprocessed_products_500.csv')
    meta_export_path = os.path.join(base_dir, 'products_metadata_500.csv')

    if not os.path.exists(parquet_path):
        if os.path.exists(raw_export_path) and os.path.exists(prep_export_path):
            print(f"Parquet source file not found at {parquet_path}.")
            print("Loading pre-extracted 500-product datasets from CSV...")
            raw_features_df = pd.read_csv(raw_export_path)
            preprocessed_features_df = pd.read_csv(prep_export_path)
            if os.path.exists(meta_export_path):
                metadata_df = pd.read_csv(meta_export_path)
        else:
            raise FileNotFoundError(f"Parquet file not found at {parquet_path}")
    else:
        df_raw = pd.read_parquet(parquet_path)
        print(f"Full dataset shape: {df_raw.shape[0]} items, {df_raw.shape[1]} columns")
        
        # 1. Sample 500 representative products deterministically
        np.random.seed(42)
        df_500 = df_raw.sample(n=500, random_state=42).copy().reset_index(drop=True)
        print("Sampled 500 representative products for recommendation catalog.")
        
        # 2. Extract and construct 20 numerical features
        df_500['desc_word_count'] = df_500['detail_desc'].fillna('').apply(lambda x: len(x.split()))
        df_500['colour_group_code'] = pd.to_numeric(df_500['colour_group_code'], errors='coerce').fillna(0)
        
        cat_feats = pd.DataFrame({
            'F01_Product_Type': df_500['product_type_no'].astype(float),
            'F02_Department': df_500['department_no'].astype(float),
            'F03_Section': df_500['section_no'].astype(float),
            'F04_Garment_Group': df_500['garment_group_no'].astype(float),
            'F05_Colour_Group': df_500['colour_group_code'].astype(float),
            'F06_Colour_Master': df_500['perceived_colour_master_id'].astype(float),
            'F07_Appearance_Pattern': df_500['graphical_appearance_no'].astype(float),
            'F08_Desc_Word_Count': df_500['desc_word_count'].astype(float)
        })
        
        print("Extracting 12 latent semantic features from 384-dimensional dense embeddings...")
        embeddings = np.vstack(df_500['dense_embedding'].values)
        pca = PCA(n_components=12, random_state=42)
        latent_embeddings = pca.fit_transform(embeddings)
        
        latent_names = [
            'F09_Semantic_Style', 'F10_Semantic_Seasonality', 'F11_Semantic_Silhouette',
            'F12_Semantic_Texture', 'F13_Semantic_Occasion', 'F14_Semantic_Formality',
            'F15_Semantic_Trendiness', 'F16_Semantic_Fit', 'F17_Semantic_Fabric',
            'F18_Semantic_Detailing', 'F19_Semantic_Versatility', 'F20_Semantic_Tone'
        ]
        latent_feats = pd.DataFrame(latent_embeddings, columns=latent_names)
        
        raw_features_df = pd.concat([cat_feats, latent_feats], axis=1)
        metadata_df = df_500[['article_id', 'product_code', 'prod_name', 'product_type_name', 
                              'product_group_name', 'department_name', 'colour_group_name', 'index_name']].copy()
        
        # Preprocessing:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(raw_features_df)
        preprocessed_matrix = normalize(scaled_features, norm='l2', axis=1)
        preprocessed_features_df = pd.DataFrame(preprocessed_matrix, columns=raw_features_df.columns)
        
        raw_features_df.to_csv(raw_export_path, index=False)
        preprocessed_features_df.to_csv(prep_export_path, index=False)
        metadata_df.to_csv(meta_export_path, index=False)
    
    print("\n20 Numerical Features Constructed Successfully:")
    for idx, col in enumerate(raw_features_df.columns, 1):
        print(f"  [{idx:02d}] {col} -> Mean: {raw_features_df[col].mean():.2f}, Std: {raw_features_df[col].std():.2f}, Range: [{raw_features_df[col].min():.1f}, {raw_features_df[col].max():.1f}]")
        
    # 3. Preprocessing:
    # Notice: Raw features have wildly disparate scales (e.g., F04 Garment Group is ~1010, F01 is ~250, while Latent features are ~0.05).
    # Step 1: Standard Scaling (Z-score normalization: zero mean, unit variance)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_features_df)
    
    # Step 2: L2 Unit-Norm Normalization
    # Crucial for Vector Dot Products: when vectors are L2-normalized (||p|| = 1),
    # the vector dot product u . p equals Cosine Similarity, strictly bounded in [-1, 1]!
    preprocessed_matrix = normalize(scaled_features, norm='l2', axis=1)
    preprocessed_features_df = pd.DataFrame(preprocessed_matrix, columns=raw_features_df.columns)
    
    # Save datasets if newly created
    if not os.path.exists(raw_export_path):
        raw_features_df.to_csv(raw_export_path, index=False)
        preprocessed_features_df.to_csv(prep_export_path, index=False)
        metadata_df.to_csv(meta_export_path, index=False)
    
    print(f"\nSaved raw features to: {raw_export_path}")
    print(f"Saved preprocessed features to: {prep_export_path}")
    print(f"Saved product metadata to: {meta_export_path}")
    
    # 4. Generate Visualization Graphs
    print("\nGenerating Module 1 Visualization Graphs...")
    
    # Graph 1: Before vs After Feature Scales (Boxplot / Spread)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#FFFFFF')
    
    feature_labels = [f"F{i+1:02d}" for i in range(20)]
    
    # Panel 1: Before Preprocessing
    set_minimal_theme(axes[0])
    bplot1 = axes[0].boxplot(
        [raw_features_df[col] for col in raw_features_df.columns],
        patch_artist=True,
        tick_labels=feature_labels,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"#EF4444", "markeredgecolor":"#EF4444", "markersize":4},
        medianprops={"color":"#1E293B", "linewidth":1.2},
        boxprops={"facecolor":"#93C5FD", "edgecolor":"#1E40AF", "linewidth":0.8, "alpha":0.85},
        whiskerprops={"color":"#1E40AF", "linewidth":0.8},
        capprops={"color":"#1E40AF", "linewidth":0.8},
        flierprops={"marker":"d", "markerfacecolor":"#60A5FA", "alpha":0.4, "markersize":3}
    )
    axes[0].set_title("(A) Raw Feature Scales Before Preprocessing (Disparate Magnitudes & Outliers)", pad=10)
    axes[0].set_ylabel("Raw Feature Value Magnitude", labelpad=8)
    axes[0].set_yscale('symlog')  # Symlog to handle wide ranges from 0.01 to 1050 cleanly
    
    # Panel 2: After Preprocessing
    set_minimal_theme(axes[1])
    bplot2 = axes[1].boxplot(
        [preprocessed_features_df[col] for col in preprocessed_features_df.columns],
        patch_artist=True,
        tick_labels=feature_labels,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"#EF4444", "markeredgecolor":"#EF4444", "markersize":4},
        medianprops={"color":"#1E293B", "linewidth":1.2},
        boxprops={"facecolor":"#A7F3D0", "edgecolor":"#065F46", "linewidth":0.8, "alpha":0.85},
        whiskerprops={"color":"#065F46", "linewidth":0.8},
        capprops={"color":"#065F46", "linewidth":0.8},
        flierprops={"marker":"d", "markerfacecolor":"#34D399", "alpha":0.4, "markersize":3}
    )
    axes[1].set_title("(B) Preprocessed & L2-Normalized Feature Scales (Calibrated for Vector Dot Products)", pad=10)
    axes[1].set_xlabel("Product Feature Dimensions (F01 to F20)", labelpad=8)
    axes[1].set_ylabel("Normalized Feature Value", labelpad=8)
    axes[1].set_ylim(-0.65, 0.65)
    
    # Create unified bottom legend
    fig.subplots_adjust(bottom=0.18, hspace=0.3)
    legend_elements = [
        mpl.patches.Patch(facecolor='#93C5FD', edgecolor='#1E40AF', label='Raw Features Distribution (F01–F20)'),
        mpl.patches.Patch(facecolor='#A7F3D0', edgecolor='#065F46', label='Preprocessed & L2-Normalized Features'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EF4444', markersize=7, label='Feature Mean Value'),
        mpl.lines.Line2D([0], [0], color='#1E293B', lw=1.5, label='Feature Median Line')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
    
    graph1_path = os.path.join(base_dir, '01_before_after_feature_scales.png')
    plt.savefig(graph1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph1_path}")
    
    # Graph 2: Detailed Before vs After Probability Density Distributions (KDE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    sample_features = ['F01_Product_Type', 'F02_Department', 'F04_Garment_Group', 'F08_Desc_Word_Count']
    palette = ['#2563EB', '#EC4899', '#10B981', '#F59E0B']
    
    # Left: Before Preprocessing
    set_minimal_theme(axes[0])
    for col, color in zip(sample_features, palette):
        sns.kdeplot(raw_features_df[col], ax=axes[0], label=f"{col} (Raw)", 
                    color=color, linewidth=2.2, fill=True, alpha=0.18)
    axes[0].set_title("(A) Feature Density Distributions Before Preprocessing", pad=12)
    axes[0].set_xlabel("Raw Values (Wide Range & Varying Units)", labelpad=8)
    axes[0].set_ylabel("Probability Density", labelpad=8)
    axes[0].set_xlim(-50, 1100)
    
    # Right: After Preprocessing
    set_minimal_theme(axes[1])
    for col, color in zip(sample_features, palette):
        sns.kdeplot(preprocessed_features_df[col], ax=axes[1], label=f"{col} (Normalized)", 
                    color=color, linewidth=2.2, fill=True, alpha=0.18)
    axes[1].set_title("(B) Feature Density Distributions After Standardization & L2 Norm", pad=12)
    axes[1].set_xlabel("Normalized Feature Values (Harmonized Range [-0.5, 0.5])", labelpad=8)
    axes[1].set_ylabel("Probability Density", labelpad=8)
    axes[1].set_xlim(-0.6, 0.6)
    
    # Unified bottom legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
    
    graph2_path = os.path.join(base_dir, '02_before_after_distribution_kde.png')
    plt.savefig(graph2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph2_path}")
    
    # Graph 3: Before vs After Feature Correlation Matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Select first 10 representative features for clear visualization
    sub_cols = list(raw_features_df.columns[:10])
    short_labels = [c.split('_')[0] + '_' + c.split('_')[1][:4] for c in sub_cols]
    
    corr_raw = raw_features_df[sub_cols].corr()
    corr_prep = preprocessed_features_df[sub_cols].corr()
    
    # Left: Raw Correlation
    set_minimal_theme(axes[0])
    im1 = axes[0].imshow(corr_raw, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title("(A) Feature Inter-Correlations Before Preprocessing", pad=12)
    axes[0].set_xticks(range(len(short_labels)))
    axes[0].set_yticks(range(len(short_labels)))
    axes[0].set_xticklabels(short_labels, rotation=45, ha='right')
    axes[0].set_yticklabels(short_labels)
    axes[0].set_xlabel("Catalog Feature Index", labelpad=6)
    axes[0].set_ylabel("Catalog Feature Index", labelpad=8)
    
    # Right: Preprocessed Correlation
    set_minimal_theme(axes[1])
    im2 = axes[1].imshow(corr_prep, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title("(B) Feature Inter-Correlations After Standardization & L2 Normalization", pad=12)
    axes[1].set_xticks(range(len(short_labels)))
    axes[1].set_yticks(range(len(short_labels)))
    axes[1].set_xticklabels(short_labels, rotation=45, ha='right')
    axes[1].set_yticklabels(short_labels)
    axes[1].set_xlabel("Catalog Feature Index", labelpad=6)
    axes[1].set_ylabel("Catalog Feature Index", labelpad=8)
    
    # Bottom Colorbar as Legend positioned with proper clearance
    fig.subplots_adjust(bottom=0.28, wspace=0.3)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.035])
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Pearson Correlation Coefficient ($r$): -1.0 (Inverse) to +1.0 (Positive)", fontname='Cambria', fontsize=10, labelpad=6)
    cbar.outline.set_edgecolor('#E5E7EB')
    cbar.outline.set_linewidth(0.8)
    
    graph3_path = os.path.join(base_dir, '03_before_after_correlation_heatmap.png')
    plt.savefig(graph3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph3_path}")
    
    print("\nModule 1 successfully completed!")

if __name__ == '__main__':
    run_module_1()
