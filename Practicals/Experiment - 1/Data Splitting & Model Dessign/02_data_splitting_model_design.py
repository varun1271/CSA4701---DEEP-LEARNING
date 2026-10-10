"""
Module 2: Data Splitting & Model Design
Scenario: An online shopping platform represents 500 products using 20 numerical features each.
Explaining how vector dot products and matrix multiplication calculate similarity scores and recommend relevant products.

This script:
1. Loads the preprocessed 500 products (each with 20 normalized numerical features).
2. Simulates realistic customer interaction behavior (100 customers with historical purchases/clicks):
   - Generates Customer Preference Matrix U (100 x 20).
   - Splits interactions into 70% Train, 15% Validation, and 15% Test.
3. Formulates the Recommendation Model Architectures:
   - Vector Dot Product Model: Pairwise dot product score(u, p) = sum(u_k * p_k).
   - Matrix Multiplication Model: Full catalog tensor multiplication S = U @ P.T (100 x 500).
   - Product-to-Product Similarity Matrix: S_item = P @ P.T (500 x 500).
4. Exports split datasets and preference matrices.
5. Generates high-quality, professional, colorful visualization graphs:
   - Font: Cambria
   - Minimalist theme with light grid
   - Clear X and Y labels with well-numbered scales
   - Legend positioned cleanly at the bottom
"""

import os
import numpy as np
import pandas as pd
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

def run_module_2():
    module2_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(module2_dir)
    prep_csv_path = os.path.join(base_dir, 'Data Collection & Preprocessing', 'preprocessed_products_500.csv')
    meta_csv_path = os.path.join(base_dir, 'Data Collection & Preprocessing', 'products_metadata_500.csv')
    
    print("=" * 70)
    print("MODULE 2: DATA SPLITTING & MODEL DESIGN")
    print("=" * 70)
    
    if not os.path.exists(prep_csv_path):
        raise FileNotFoundError(f"Preprocessed dataset not found at {prep_csv_path}. Run Module 1 first.")
        
    df_prep = pd.read_csv(prep_csv_path)
    df_meta = pd.read_csv(meta_csv_path)
    
    P = df_prep.values  # Product Matrix P in R^(500 x 20)
    N_products, D_features = P.shape
    print(f"Product Catalog Matrix P: Shape = ({N_products}, {D_features})")
    
    # 1. Simulate 100 Customers with distinct shopping affinities
    # Each customer has preferences formed by their interacted items + personalized noise
    np.random.seed(42)
    M_customers = 100
    
    # Create 5 distinct shopping persona centers (e.g. Formal, Casual, Sporty, Trendy, Classic)
    persona_centers = np.random.randn(5, D_features)
    persona_centers = persona_centers / np.linalg.norm(persona_centers, axis=1, keepdims=True)
    
    customer_personas = np.random.choice(5, size=M_customers)
    customer_raw_vectors = persona_centers[customer_personas] + np.random.randn(M_customers, D_features) * 0.25
    # L2-normalize customer vectors
    U = customer_raw_vectors / np.linalg.norm(customer_raw_vectors, axis=1, keepdims=True)
    print(f"Customer Preference Matrix U: Shape = ({M_customers}, {D_features})")
    
    # 2. Simulate User-Product Historical Interactions (Affinity ground truth)
    # Ground truth affinity = dot product + interaction noise
    all_affinities = np.dot(U, P.T)  # (100 x 500)
    
    interactions = []
    for u_idx in range(M_customers):
        # Pick top 20 items the customer had affinity towards
        top_items = np.argsort(-all_affinities[u_idx])[:20]
        # Interacted items with timestamp / interaction weights
        for rank, p_idx in enumerate(top_items):
            interactions.append({
                'customer_id': u_idx,
                'product_id': p_idx,
                'affinity_score': float(all_affinities[u_idx, p_idx]),
                'interaction_rank': rank + 1
            })
            
    df_interactions = pd.DataFrame(interactions)
    print(f"Total Customer Interactions Generated: {len(df_interactions)}")
    
    # 3. Data Splitting: Train (70%), Validation (15%), Test (15%)
    # Standard Leave-K-Out interaction splitting per customer
    train_list, val_list, test_list = [], [], []
    for u_idx, group in df_interactions.groupby('customer_id'):
        n_items = len(group)
        n_train = int(n_items * 0.70)
        n_val = int(n_items * 0.15)
        
        # Shuffle interactions per user deterministically
        shuffled = group.sample(frac=1.0, random_state=42 + u_idx).reset_index(drop=True)
        train_list.append(shuffled.iloc[:n_train])
        val_list.append(shuffled.iloc[n_train:n_train + n_val])
        test_list.append(shuffled.iloc[n_train + n_val:])
        
    df_train = pd.concat(train_list).reset_index(drop=True)
    df_val = pd.concat(val_list).reset_index(drop=True)
    df_test = pd.concat(test_list).reset_index(drop=True)
    
    print(f"\nData Splitting Complete:")
    print(f"  Train Set      : {len(df_train)} interactions ({len(df_train)/len(df_interactions)*100:.1f}%)")
    print(f"  Validation Set : {len(df_val)} interactions ({len(df_val)/len(df_interactions)*100:.1f}%)")
    print(f"  Test Set       : {len(df_test)} interactions ({len(df_test)/len(df_interactions)*100:.1f}%)")
    
    # Save datasets
    df_train.to_csv(os.path.join(module2_dir, 'train_interactions.csv'), index=False)
    df_val.to_csv(os.path.join(module2_dir, 'val_interactions.csv'), index=False)
    df_test.to_csv(os.path.join(module2_dir, 'test_interactions.csv'), index=False)
    np.save(os.path.join(module2_dir, 'customer_preference_matrix.npy'), U)
    np.save(os.path.join(module2_dir, 'product_catalog_matrix.npy'), P)
    
    # 4. Compute Product-Product Similarity Matrix S_prod = P @ P.T
    print("\nComputing Product-to-Product Similarity Matrix S_prod = P @ P.T...")
    S_prod = np.dot(P, P.T)  # (500, 500)
    print(f"S_prod Matrix Shape: {S_prod.shape}, Range: [{S_prod.min():.3f}, {S_prod.max():.3f}]")
    
    # 5. Compute Customer-Product Similarity Matrix S = U @ P.T
    print("Computing Customer-to-Product Similarity Matrix S = U @ P.T...")
    S_user_prod = np.dot(U, P.T)  # (100, 500)
    print(f"S_user_prod Matrix Shape: {S_user_prod.shape}, Range: [{S_user_prod.min():.3f}, {S_user_prod.max():.3f}]")
    
    # 6. Generate Module 2 Graphs
    print("\nGenerating Module 2 Visualization Graphs...")
    
    # Graph 4: Data Splitting Proportions & Split Interaction Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Donut Chart for Split Proportions
    split_sizes = [len(df_train), len(df_val), len(df_test)]
    split_labels = ['Training Set (70%)', 'Validation Set (15%)', 'Test Set (15%)']
    split_colors = ['#2563EB', '#F59E0B', '#10B981']
    
    axes[0].set_facecolor('#FFFFFF')
    wedges, texts, autotexts = axes[0].pie(
        split_sizes, 
        labels=None,
        autopct='%1.1f%%', 
        pctdistance=0.75,
        startangle=140, 
        colors=split_colors,
        wedgeprops=dict(width=0.45, edgecolor='#FFFFFF', linewidth=2)
    )
    for autotext in autotexts:
        autotext.set_fontname('Cambria')
        autotext.set_fontsize(11)
        autotext.set_color('#FFFFFF')
        autotext.set_weight('bold')
    axes[0].set_title("(A) Interaction Data Partition Proportions", pad=12)
    
    # Subplot B: Affinity Score Distribution Across Splits
    set_minimal_theme(axes[1])
    sns.kdeplot(df_train['affinity_score'], ax=axes[1], color='#2563EB', label='Train Set (1,400 records)', linewidth=2.2, fill=True, alpha=0.15)
    sns.kdeplot(df_val['affinity_score'], ax=axes[1], color='#F59E0B', label='Validation Set (300 records)', linewidth=2.2, fill=True, alpha=0.15)
    sns.kdeplot(df_test['affinity_score'], ax=axes[1], color='#10B981', label='Test Set (300 records)', linewidth=2.2, fill=True, alpha=0.15)
    axes[1].set_title("(B) Affinity Score Distribution Consistency Across Splits", pad=12)
    axes[1].set_xlabel(r"Cosine Similarity Score ($\mathbf{u} \cdot \mathbf{p}$)", labelpad=8)
    axes[1].set_ylabel("Probability Density", labelpad=8)
    axes[1].set_xlim(-0.2, 1.0)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    legend_handles = [
        mpl.patches.Patch(facecolor='#2563EB', edgecolor='#1E40AF', label=f'Train Split ({len(df_train)} items, 70%)'),
        mpl.patches.Patch(facecolor='#F59E0B', edgecolor='#B45309', label=f'Validation Split ({len(df_val)} items, 15%)'),
        mpl.patches.Patch(facecolor='#10B981', edgecolor='#047857', label=f'Test Split ({len(df_test)} items, 15%)')
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=3, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
    
    graph4_path = os.path.join(module2_dir, '04_data_splitting_proportions.png')
    plt.savefig(graph4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph4_path}")
    
    # Graph 5: Product Similarity Matrix Heatmap (P @ P.T)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    fig.patch.set_facecolor('#FFFFFF')
    set_minimal_theme(ax)
    
    # Subsample 100 products for clear visual representation of heatmap structure
    sub_P = P[:100]
    S_sub = np.dot(sub_P, sub_P.T)
    
    im = ax.imshow(S_sub, cmap='viridis', vmin=-0.4, vmax=1.0, aspect='equal')
    ax.set_title(r"Product-to-Product Similarity Matrix Heatmap ($\mathbf{S}_{prod} = \mathbf{P} \mathbf{P}^T$)", pad=14)
    ax.set_xlabel(r"Catalog Product Index ($j \in \{0, \dots, 99\}$)", labelpad=8)
    ax.set_ylabel(r"Catalog Product Index ($i \in \{0, \dots, 99\}$)", labelpad=8)
    
    # Colorbar at the bottom
    fig.subplots_adjust(bottom=0.20)
    cbar_ax = fig.add_axes([0.22, 0.07, 0.56, 0.035])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r"Dot Product Similarity Score: $\mathbf{p}_i \cdot \mathbf{p}_j = \cos(\theta_{i,j})$ (1.0 = Identical, -0.4 = Dissimilar)", 
                   fontname='Cambria', fontsize=10, labelpad=6)
    cbar.outline.set_edgecolor('#E5E7EB')
    cbar.outline.set_linewidth(0.8)
    
    graph5_path = os.path.join(module2_dir, '05_product_similarity_matrix_heatmap.png')
    plt.savefig(graph5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph5_path}")
    
    # Graph 6: Model Design: Vector Dot Product vs Matrix Multiplication Architecture
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Left: Histogram of all 50,000 Pairwise Dot Products (U x P.T)
    set_minimal_theme(axes[0])
    flat_scores = S_user_prod.flatten()
    axes[0].hist(flat_scores, bins=50, color='#6366F1', edgecolor='#4338CA', alpha=0.75, density=True)
    axes[0].axvline(np.mean(flat_scores), color='#EF4444', linestyle='--', linewidth=2, label=f'Catalog Mean ({np.mean(flat_scores):.2f})')
    axes[0].axvline(np.percentile(flat_scores, 95), color='#10B981', linestyle='-', linewidth=2, label=f'95th Percentile Threshold ({np.percentile(flat_scores, 95):.2f})')
    axes[0].set_title(r"(A) Catalog Similarity Score Distribution ($M \times N = 50,000$ Pairs)", pad=12)
    axes[0].set_xlabel(r"Similarity Score ($\mathbf{u} \cdot \mathbf{p}$)", labelpad=8)
    axes[0].set_ylabel("Probability Density", labelpad=8)
    axes[0].set_xlim(-0.6, 1.0)
    
    # Right: Dimensional Flow & Matrix Operation Comparison
    set_minimal_theme(axes[1])
    # Bar plot illustrating FLOPs and tensor dimensions
    ops_labels = ['Scalar Multiplications\nper Customer', 'Operations for\n100 Customers (Loops)', 'Matrix Mult FLOPs\n' + r'$\mathbf{U}_{(100 \times 20)} \mathbf{P}^T_{(20 \times 500)}$']
    ops_values = [500 * 20, 100 * 500 * 20, 100 * 500 * 20]
    bar_colors = ['#3B82F6', '#EC4899', '#10B981']
    
    bars = axes[1].bar(ops_labels, ops_values, color=bar_colors, width=0.55, edgecolor='#1E293B', linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(f'{h:,} ops',
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontname='Cambria', fontsize=10, weight='bold')
    axes[1].set_title("(B) Computational Operations Profile in Matrix Formulation", pad=12)
    axes[1].set_ylabel("Arithmetic Multiply-Add Operations", labelpad=8)
    axes[1].set_ylim(0, 1.3e6)
    axes[1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    leg_handles = [
        mpl.lines.Line2D([0], [0], color='#EF4444', linestyle='--', lw=2, label='Mean Catalog Similarity (0.00)'),
        mpl.lines.Line2D([0], [0], color='#10B981', linestyle='-', lw=2, label='Top-K Recommendation Boundary (0.37)'),
        mpl.patches.Patch(facecolor='#3B82F6', edgecolor='#1E40AF', label='Per-Customer Ops (10k)'),
        mpl.patches.Patch(facecolor='#EC4899', edgecolor='#9D174D', label='Loop Operations (1M ops)'),
        mpl.patches.Patch(facecolor='#10B981', edgecolor='#047857', label='Vectorized Matrix Ops (1M FLOPs)')
    ]
    fig.legend(handles=leg_handles, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=5, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
    
    graph6_path = os.path.join(module2_dir, '06_vector_dot_vs_matrix_mult_architecture.png')
    plt.savefig(graph6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph6_path}")
    
    print("\nModule 2 successfully completed!")

if __name__ == '__main__':
    run_module_2()
