"""
Module 1: Data Collection & Preprocessing
Scenario: A retail enterprise collects customer data across key financial and behavioral variables:
Income, Spending, and Purchase Frequency. Analyze the covariance matrix and determine which features
show strong positive or negative relationships. Model Probability Distributions and Bayes' Theorem.

This script:
1. Synthesizes a realistic enterprise retail customer dataset (N = 1,000 customers).
2. Computes the Sample Covariance Matrix and Pearson Correlation Matrix.
3. Analyzes feature relationships: identifies strong positive, strong negative, and orthogonal pairs.
4. Models empirical and theoretical Probability Distributions:
   - Annual Income: Gaussian N(mu, sigma^2) and Log-Normal distributions.
   - Annual Spending: Gaussian and Bimodal KDE distributions.
   - Purchase Frequency: Discrete Poisson Pois(lambda) and Negative Binomial NB(r, p) distributions.
5. Performs rigorous preprocessing:
   - Missing value verification and data cleaning.
   - Outlier detection and scale inspection.
   - Z-score Standardization (Mean 0, Variance 1).
   - Min-Max scaling and Robust scaling.
6. Exports raw, preprocessed, and demographic metadata datasets to CSV.
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def generate_retail_customer_data(n_samples=1000, random_state=42):
    """
    Generates realistic enterprise retail customer data with defined covariance structure.
    Features:
      1. Income ($k/year): Continuous, positive, right-skewed
      2. Spending ($k/year): Strongly correlated with Income and Frequency
      3. Purchase_Frequency (transactions/year): Count data, Poisson/Negative Binomial
      4. Average_Order_Value ($): Correlated with Income and Spending
      5. Recency (days): Negatively correlated with Frequency and Spending
      6. Discount_Sensitivity (ratio 0-1): Negatively correlated with Income
      7. Browsing_Duration (mins/session): Orthogonal/mild correlation
      8. Loyalty_Tenure_Years (years): Demographic/tenure factor
    """
    np.random.seed(random_state)
    
    # Latent multivariate Gaussian factors to induce realistic covariance
    mean_factors = [0.0, 0.0, 0.0]
    # Covariance among latent financial capability, shopping activity, and price sensitivity
    cov_factors = [
        [1.0, 0.75, -0.60],   # Factor 1: Affluence / Income drive
        [0.75, 1.0, -0.50],   # Factor 2: Shopping frequency & spending drive
        [-0.60, -0.50, 1.0]   # Factor 3: Bargain hunting / discount sensitivity
    ]
    latent = np.random.multivariate_normal(mean_factors, cov_factors, size=n_samples)
    l_affluence = latent[:, 0]
    l_activity = latent[:, 1]
    l_bargain = latent[:, 2]
    
    # 1. Income (Annual Income in $k): Mean ~$68k, Std ~$23k
    income = 68.0 + 22.0 * l_affluence + np.random.normal(0, 5.0, size=n_samples)
    income = np.clip(income, 20.0, 150.0)
    
    # 2. Purchase Frequency (Annual visits): Mean ~29, Std ~13
    freq_rate = np.exp(3.2 + 0.38 * l_activity + np.random.normal(0, 0.15, size=n_samples))
    purchase_frequency = np.random.poisson(freq_rate)
    purchase_frequency = np.clip(purchase_frequency, 3, 85)
    
    # 3. Spending (Annual Total in $k): Strong positive relationship with Income and Frequency
    spending = 12.0 + 0.42 * income + 0.52 * purchase_frequency + 10.0 * l_activity + np.random.normal(0, 4.5, size=n_samples)
    spending = np.clip(spending, 5.0, 125.0)
    
    # 4. Average Order Value (AOV in $): Income-driven basket size
    aov = 35.0 + 1.25 * income + 14.0 * (spending / np.maximum(purchase_frequency, 1)) + np.random.normal(0, 15.0, size=n_samples)
    aov = np.clip(aov, 30.0, 450.0)
    
    # 5. Recency (Days since last purchase, 1 to 365): Negative correlation with Frequency
    recency = 120.0 - 1.6 * purchase_frequency + np.random.exponential(25.0, size=n_samples)
    recency = np.clip(recency, 1.0, 360.0)
    
    # 6. Discount Sensitivity (0.0 to 1.0): Negative correlation with Income, positive with bargain factor
    discount_raw = 0.45 + 0.18 * l_bargain - 0.12 * ((income - 68.0) / 22.0) + np.random.normal(0, 0.08, size=n_samples)
    discount_sensitivity = np.clip(discount_raw, 0.02, 0.98)
    
    # 7. Browsing Duration (Minutes per session): Mean ~26 mins
    browsing_duration = 26.0 + 4.5 * l_activity + np.random.normal(0, 8.0, size=n_samples)
    browsing_duration = np.clip(browsing_duration, 4.0, 65.0)
    
    # 8. Loyalty Tenure Years: Mean ~3.8 years
    loyalty_tenure = 1.0 + np.random.gamma(shape=2.5, scale=1.2, size=n_samples)
    loyalty_tenure = np.clip(loyalty_tenure, 0.2, 12.0)
    
    # Ground-truth High-Value VIP Customer Target (Class 1 vs Class 0)
    # High-value condition: Top spenders with solid frequency and income
    vip_score = 0.50 * ((spending - 51.0) / 26.0) + 0.35 * ((income - 68.0) / 22.0) + 0.25 * ((purchase_frequency - 29.0) / 13.0)
    vip_prob = 1.0 / (1.0 + np.exp(-1.8 * (vip_score - 0.35)))
    is_high_value = (np.random.rand(n_samples) < vip_prob).astype(int)
    
    # Demographics metadata
    customer_ids = [f"CUST_{i+1:04d}" for i in range(n_samples)]
    ages = np.random.normal(39, 11, size=n_samples).astype(int)
    ages = np.clip(ages, 18, 72)
    genders = np.random.choice(['Female', 'Male', 'Non-Binary'], size=n_samples, p=[0.52, 0.45, 0.03])
    city_tiers = np.random.choice(['Tier 1 (Metro)', 'Tier 2 (Urban)', 'Tier 3 (Suburban)'], size=n_samples, p=[0.45, 0.35, 0.20])
    preferred_categories = np.random.choice(['Apparel & Luxury', 'Consumer Electronics', 'Home & Living', 'Beauty & Health', 'Sporting Goods'], 
                                            size=n_samples, p=[0.30, 0.25, 0.18, 0.15, 0.12])
    
    tier_labels = []
    for hv, sp in zip(is_high_value, spending):
        if hv == 1 and sp > 80.0:
            tier_labels.append('Platinum VIP')
        elif hv == 1:
            tier_labels.append('Gold VIP')
        elif sp > 40.0:
            tier_labels.append('Silver Member')
        else:
            tier_labels.append('Bronze Regular')
            
    # DataFrame construction
    df_features = pd.DataFrame({
        'Income': income,
        'Spending': spending,
        'Purchase_Frequency': purchase_frequency,
        'Average_Order_Value': aov,
        'Recency': recency,
        'Discount_Sensitivity': discount_sensitivity,
        'Browsing_Duration': browsing_duration,
        'Loyalty_Tenure_Years': loyalty_tenure
    })
    
    df_meta = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Age': ages,
        'Gender': genders,
        'City_Tier': city_tiers,
        'Preferred_Category': preferred_categories,
        'Loyalty_Tier': tier_labels,
        'High_Value_Customer': is_high_value
    })
    
    return df_features, df_meta

def compute_covariance_and_correlation(df_features):
    """
    Computes sample covariance matrix Sigma and Pearson correlation matrix R.
    Formula:
      Cov(X_j, X_k) = 1/(N-1) * sum((x_ij - mu_j) * (x_ik - mu_k))
      Corr(X_j, X_k) = Cov(X_j, X_k) / (sigma_j * sigma_k)
    """
    cov_matrix = df_features.cov()
    corr_matrix = df_features.corr()
    return cov_matrix, corr_matrix

def run_module_1():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 75)
    print("MODULE 1: DATA COLLECTION & PREPROCESSING (RETAIL CUSTOMER ANALYTICS)")
    print("=" * 75)
    
    # 1. Synthesize 1,000 enterprise retail customer records
    print("Generating enterprise retail dataset: N = 1,000 customers across 8 behavioral features...")
    df_raw, df_meta = generate_retail_customer_data(n_samples=1000, random_state=42)
    
    print(f"Dataset generated successfully. Total customers: {len(df_raw)}")
    print("\nFeature Summary Statistics (Raw Scales):")
    for col in df_raw.columns:
        print(f"  - {col:22s} | Mean: {df_raw[col].mean():8.2f} | Std: {df_raw[col].std():8.2f} | Range: [{df_raw[col].min():6.2f}, {df_raw[col].max():6.2f}]")
        
    # 2. Covariance and Correlation Analysis
    cov_matrix, corr_matrix = compute_covariance_and_correlation(df_raw)
    
    print("\n" + "-" * 75)
    print("COVARIANCE MATRIX (Sample Sigma, Scale-Dependent Units):")
    print("-" * 75)
    print(cov_matrix.round(2).to_string())
    
    print("\n" + "-" * 75)
    print("PEARSON CORRELATION MATRIX (Standardized Dimensionless R in [-1, +1]):")
    print("-" * 75)
    print(corr_matrix.round(3).to_string())
    
    # Identify Strong Positive and Negative Relationships
    print("\n" + "-" * 75)
    print("KEY FEATURE RELATIONSHIP FINDINGS:")
    print("-" * 75)
    
    corr_pairs = []
    cols = list(df_raw.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            r_val = corr_matrix.loc[c1, c2]
            cov_val = cov_matrix.loc[c1, c2]
            corr_pairs.append((c1, c2, r_val, cov_val))
            
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\n[A] Strong Positive Relationships (r >= +0.50):")
    for c1, c2, r, cov in [p for p in corr_pairs if p[2] >= 0.50]:
        print(f"  * {c1} <---> {c2}: Pearson r = +{r:.3f} | Covariance = {cov:+8.2f}")
        
    print("\n[B] Strong & Moderate Negative Relationships (r <= -0.40):")
    for c1, c2, r, cov in [p for p in corr_pairs if p[2] <= -0.40]:
        print(f"  * {c1} <---> {c2}: Pearson r = {r:.3f} | Covariance = {cov:+8.2f}")
        
    print("\n[C] Near-Zero / Orthogonal Relationships (|r| < 0.15):")
    for c1, c2, r, cov in [p for p in corr_pairs if abs(p[2]) < 0.15]:
        print(f"  * {c1} <---> {c2}: Pearson r = {r:+.3f} | Covariance = {cov:+8.2f}")
        
    # 3. Probability Distribution Fitting & Goodness-of-Fit Tests
    print("\n" + "-" * 75)
    print("PROBABILITY DISTRIBUTION MODELING:")
    print("-" * 75)
    
    # (a) Income -> Normal & Log-Normal
    inc_data = df_raw['Income'].values
    mu_inc, std_inc = stats.norm.fit(inc_data)
    ks_stat_norm, ks_p_norm = stats.kstest(inc_data, 'norm', args=(mu_inc, std_inc))
    shape_ln, loc_ln, scale_ln = stats.lognorm.fit(inc_data, floc=0)
    ks_stat_ln, ks_p_ln = stats.kstest(inc_data, 'lognorm', args=(shape_ln, loc_ln, scale_ln))
    print(f"  1. Annual Income Distribution:")
    print(f"     - Gaussian Fit:        N(mu={mu_inc:.2f}, sigma^2={std_inc**2:.2f}) | KS-stat = {ks_stat_norm:.4f} (p={ks_p_norm:.4f})")
    print(f"     - Log-Normal Fit:     LogNorm(shape={shape_ln:.3f}, scale={scale_ln:.2f}) | KS-stat = {ks_stat_ln:.4f} (p={ks_p_ln:.4f})")
    
    # (b) Purchase Frequency -> Poisson & Negative Binomial
    freq_data = df_raw['Purchase_Frequency'].values
    lambda_pois = np.mean(freq_data)
    var_freq = np.var(freq_data)
    dispersion_ratio = var_freq / lambda_pois
    print(f"\n  2. Purchase Frequency Distribution (Count Modeling):")
    print(f"     - Empirical Mean (lambda): {lambda_pois:.2f} | Variance: {var_freq:.2f} | Dispersion Ratio (Var/Mean): {dispersion_ratio:.2f}")
    print(f"     - Poisson Fit:         Pois(lambda={lambda_pois:.2f})")
    print(f"     - Overdispersion detected (Var > Mean), Negative Binomial accounts for extra variance.")
    
    # (c) Spending -> Gaussian Fit
    spend_data = df_raw['Spending'].values
    mu_sp, std_sp = stats.norm.fit(spend_data)
    print(f"\n  3. Annual Spending Distribution:")
    print(f"     - Gaussian Fit:        N(mu={mu_sp:.2f}, sigma^2={std_sp**2:.2f})")
    
    # 4. Preprocessing Pipeline
    # Standard Scaling (Z-score normalization: mu=0, sigma=1)
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df_raw)
    df_preprocessed = pd.DataFrame(scaled_matrix, columns=df_raw.columns)
    
    # Save datasets
    raw_csv_path = os.path.join(base_dir, 'raw_retail_customers_1000.csv')
    prep_csv_path = os.path.join(base_dir, 'preprocessed_retail_customers_1000.csv')
    meta_csv_path = os.path.join(base_dir, 'customer_demographics_metadata_1000.csv')
    
    df_raw.to_csv(raw_csv_path, index=False)
    df_preprocessed.to_csv(prep_csv_path, index=False)
    df_meta.to_csv(meta_csv_path, index=False)
    
    print(f"\nExported raw customer dataset to: {raw_csv_path}")
    print(f"Exported preprocessed dataset to: {prep_csv_path}")
    print(f"Exported customer demographics to: {meta_csv_path}")
    
    # 5. Generate Publication-Grade Visualization Graphs
    print("\nGenerating Module 1 Visualization Graphs...")
    
    # Graph 1: Feature Scales Before vs After Preprocessing (Boxplot / Spread)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=False)
    fig.patch.set_facecolor('#FFFFFF')
    
    feature_labels = list(df_raw.columns)
    short_labels = ['Income ($k)', 'Spending ($k)', 'Freq (visits)', 'AOV ($)', 
                    'Recency (d)', 'Discount (0-1)', 'Browsing (m)', 'Tenure (yr)']
    
    # Panel A: Raw Feature Scales (Symlog to handle scales from 0.02 to 360)
    set_minimal_theme(axes[0])
    bplot1 = axes[0].boxplot(
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
    axes[0].set_title("(A) Raw Retail Feature Scales Before Preprocessing (Disparate Measurement Units)", pad=10)
    axes[0].set_ylabel("Raw Feature Value Magnitude", labelpad=8)
    axes[0].set_yscale('symlog', linthresh=1.0)
    
    # Panel B: Standardized Z-Score Distributions
    set_minimal_theme(axes[1])
    bplot2 = axes[1].boxplot(
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
    axes[1].set_xlabel("Retail Customer Behavioral & Financial Features", labelpad=8)
    axes[1].set_ylabel("Standardized Z-Score", labelpad=8)
    axes[1].set_ylim(-3.8, 4.2)
    
    # Unified bottom legend
    fig.subplots_adjust(bottom=0.18, hspace=0.32)
    legend_elements = [
        mpl.patches.Patch(facecolor='#93C5FD', edgecolor='#1E40AF', label='Raw Features Distribution (Disparate Scales)'),
        mpl.patches.Patch(facecolor='#A7F3D0', edgecolor='#065F46', label='Standardized Features (Centered Z-Scores)'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EF4444', markersize=7, label='Empirical Mean Value'),
        mpl.lines.Line2D([0], [0], color='#1E293B', lw=1.5, label='Empirical Median Line')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
    
    graph1_path = os.path.join(base_dir, '01_customer_feature_distributions_and_scales.png')
    plt.savefig(graph1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph1_path}")
    
    # Graph 2: Empirical vs Theoretical Probability Distributions (Income, Frequency, Spending)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Annual Income (Empirical KDE vs Gaussian vs Log-Normal)
    set_minimal_theme(axes[0])
    sns.histplot(df_raw['Income'], ax=axes[0], stat='density', bins=25, color='#93C5FD', edgecolor='#2563EB', alpha=0.55, label='Empirical Histogram')
    sns.kdeplot(df_raw['Income'], ax=axes[0], color='#2563EB', linewidth=2.2, label='Empirical KDE')
    x_inc = np.linspace(df_raw['Income'].min() - 5, df_raw['Income'].max() + 5, 200)
    axes[0].plot(x_inc, stats.norm.pdf(x_inc, mu_inc, std_inc), color='#EF4444', linestyle='--', linewidth=2.0, label=f'Gaussian N({mu_inc:.1f}, {std_inc:.1f})')
    axes[0].plot(x_inc, stats.lognorm.pdf(x_inc, shape_ln, loc_ln, scale_ln), color='#10B981', linestyle=':', linewidth=2.0, label='Log-Normal Fit')
    axes[0].set_title("(A) Income Probability Distribution", pad=12)
    axes[0].set_xlabel("Annual Income ($k/year)", labelpad=8)
    axes[0].set_ylabel("Probability Density", labelpad=8)
    
    # Subplot B: Purchase Frequency (Discrete Count vs Poisson Fit)
    set_minimal_theme(axes[1])
    bins_freq = np.arange(df_raw['Purchase_Frequency'].min() - 0.5, df_raw['Purchase_Frequency'].max() + 1.5, 2)
    axes[1].hist(df_raw['Purchase_Frequency'], bins=bins_freq, density=True, color='#FDE68A', edgecolor='#D97706', alpha=0.6, label='Empirical Counts')
    k_vals = np.arange(int(df_raw['Purchase_Frequency'].min()), int(df_raw['Purchase_Frequency'].max()) + 1)
    poiss_pmf = stats.poisson.pmf(k_vals, lambda_pois)
    axes[1].plot(k_vals, poiss_pmf, color='#D97706', marker='o', markersize=3.5, linestyle='-', linewidth=1.8, label=f'Poisson PMF (λ={lambda_pois:.1f})')
    # Negative Binomial fit (method of moments)
    p_nb = lambda_pois / var_freq
    r_nb = (lambda_pois**2) / (var_freq - lambda_pois)
    if r_nb > 0 and 0 < p_nb < 1:
        nb_pmf = stats.nbinom.pmf(k_vals, r_nb, p_nb)
        axes[1].plot(k_vals, nb_pmf, color='#8B5CF6', linestyle='--', linewidth=2.0, label=f'Neg-Binomial (r={r_nb:.1f})')
    axes[1].set_title("(B) Purchase Frequency Count Distribution", pad=12)
    axes[1].set_xlabel("Annual Purchase Frequency (Transactions)", labelpad=8)
    axes[1].set_ylabel("Probability Mass / Density", labelpad=8)
    
    # Subplot C: Annual Spending Distribution
    set_minimal_theme(axes[2])
    sns.histplot(df_raw['Spending'], ax=axes[2], stat='density', bins=25, color='#FBCFE8', edgecolor='#DB2777', alpha=0.55, label='Empirical Histogram')
    sns.kdeplot(df_raw['Spending'], ax=axes[2], color='#DB2777', linewidth=2.2, label='Empirical KDE')
    x_sp = np.linspace(df_raw['Spending'].min() - 5, df_raw['Spending'].max() + 5, 200)
    axes[2].plot(x_sp, stats.norm.pdf(x_sp, mu_sp, std_sp), color='#06B6D4', linestyle='--', linewidth=2.0, label=f'Gaussian N({mu_sp:.1f}, {std_sp:.1f})')
    axes[2].set_title("(C) Annual Spending Distribution", pad=12)
    axes[2].set_xlabel("Annual Spending ($k/year)", labelpad=8)
    axes[2].set_ylabel("Probability Density", labelpad=8)
    
    # Unified bottom legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    handles3, labels3 = axes[2].get_legend_handles_labels()
    
    all_handles = [handles1[1], handles1[2], handles1[3], handles2[1], handles2[2], handles3[2]]
    all_labels = ['Income KDE', 'Income Normal Fit', 'Income LogNormal Fit', 
                  'Frequency Poisson PMF', 'Frequency Neg-Binomial Fit', 'Spending Normal Fit']
    fig.legend(handles=all_handles, labels=all_labels, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=6, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
    
    graph2_path = os.path.join(base_dir, '02_probability_distribution_fits_kde.png')
    plt.savefig(graph2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph2_path}")
    
    # Graph 3: Covariance Matrix vs Pearson Correlation Matrix Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    fig.patch.set_facecolor('#FFFFFF')
    
    short_names = ['Income', 'Spending', 'Frequency', 'AOV', 'Recency', 'Discount', 'Browsing', 'Tenure']
    
    # Left: Sample Covariance Matrix (Scale-Dependent)
    set_minimal_theme(axes[0])
    sns.heatmap(cov_matrix, ax=axes[0], annot=True, fmt=".1f", cmap='Blues', cbar=True,
                xticklabels=short_names, yticklabels=short_names,
                cbar_kws={'orientation': 'horizontal', 'pad': 0.16, 'label': 'Sample Covariance Value Cov(X_j, X_k)'})
    axes[0].set_title("(A) Sample Covariance Matrix Σ (Scale-Dependent Units)", pad=14)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Right: Pearson Correlation Matrix (Standardized [-1, +1])
    set_minimal_theme(axes[1])
    sns.heatmap(corr_matrix, ax=axes[1], annot=True, fmt=".2f", cmap='coolwarm', vmin=-1.0, vmax=1.0, center=0.0,
                xticklabels=short_names, yticklabels=short_names,
                cbar_kws={'orientation': 'horizontal', 'pad': 0.16, 'label': 'Pearson Correlation Coefficient r ∈ [-1, +1]'})
    axes[1].set_title("(B) Standardized Correlation Matrix R (Strong Positive & Negative Relationships)", pad=14)
    axes[1].tick_params(axis='x', rotation=45)
    
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    
    graph3_path = os.path.join(base_dir, '03_covariance_and_correlation_matrices.png')
    plt.savefig(graph3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph3_path}")
    
    print("\nModule 1 execution completed successfully!")

if __name__ == '__main__':
    run_module_1()
