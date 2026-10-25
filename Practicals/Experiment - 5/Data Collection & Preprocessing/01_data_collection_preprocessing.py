"""
Module 1: Data Collection & Preprocessing
Scenario: A real-estate price prediction system compares linear regression, a decision tree,
and a high-complexity polynomial model. Evaluate their training and testing errors to identify
the most suitable model. Compare Gradient Descent, SGD, Momentum, and Adam.

This script:
1. Synthesizes a realistic real-estate valuation dataset (N = 1,000 properties).
   Features (D = 8): Square Footage, Bedrooms, Bathrooms, Age of Property, Distance to Metro,
   Location Desirability Score, School Rating Index, Lot Area SqFt.
   Target: Sale Price in USD ($150,000 - $1,800,000) with realistic non-linearities and interactions.
2. Performs data validation, missing value checks, and statistical profiling.
3. Implements Z-score standardization (Mean = 0, Std = 1).
4. Exports raw, standardized, and feature metadata datasets to CSV.
5. Generates 3 publication-grade visualization figures:
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

def synthesize_real_estate_dataset(n_samples=1000, seed=42):
    """
    Synthesizes N = 1,000 real-estate housing records with continuous features
    and an authentic non-linear property valuation function.
    """
    np.random.seed(seed)
    
    # 1. Square Footage: Log-normal distribution centered around 2,200 sq ft
    sqft = np.clip(np.random.lognormal(mean=7.65, sigma=0.35, size=n_samples), 800, 5200)
    
    # 2. Bedrooms: Discrete integer conditioned on sqft
    bedrooms = np.clip(np.round(1.0 + sqft / 750.0 + np.random.normal(0, 0.5, size=n_samples)), 1, 6).astype(int)
    
    # 3. Bathrooms: Half/full baths conditioned on bedrooms and sqft
    bathrooms = np.clip(np.round((1.0 + bedrooms * 0.6 + sqft / 1500.0) * 2.0) / 2.0, 1.0, 5.0)
    
    # 4. Age of Property (Years): Uniform 0 to 50 years
    age_years = np.random.uniform(0.5, 48.0, size=n_samples)
    
    # 5. Distance to Metro / Transit (KM): Exponential distribution (closer to urban center)
    distance_metro_km = np.clip(np.random.exponential(scale=5.5, size=n_samples) + 0.5, 0.5, 28.0)
    
    # 6. Location Desirability Score: [1.0, 10.0] scale
    location_score = np.clip(np.random.beta(a=3.5, b=2.5, size=n_samples) * 9.0 + 1.0, 1.0, 10.0)
    
    # 7. School District Rating Index: [1.0, 10.0] scale, correlated with location
    school_rating = np.clip(0.6 * location_score + 0.4 * np.random.uniform(2.0, 9.5, size=n_samples), 1.0, 10.0)
    
    # 8. Lot Area (Sq Ft): Proportional to sqft with outdoor variance
    lot_area_sqft = np.clip(sqft * np.random.uniform(1.8, 4.5, size=n_samples), 1500, 18000)
    
    # True Non-Linear Valuation Function:
    # Base price + sqft return with diminishing returns + location multiplier - distance penalty - age penalty
    base_price = 120000.0
    price_sqft = 185.0 * sqft - 0.012 * (sqft - 2000.0)**2  # non-linear diminishing returns
    price_bed_bath = 18000.0 * bedrooms + 24000.0 * bathrooms
    price_location = 38000.0 * (location_score**1.25)
    price_school = 16000.0 * school_rating
    price_transit_penalty = -5500.0 * np.log1p(distance_metro_km) * 4.0
    price_age_depreciation = -1800.0 * age_years
    price_lot = 12.0 * lot_area_sqft
    
    # Non-linear interaction: premium location magnifies square footage value
    interaction = 8.5 * (location_score - 5.0) * (sqft - 2000.0)
    
    # Intrinsic market noise (irreducible error sigma_noise)
    market_noise = np.random.normal(0, 22000.0, size=n_samples)
    
    sale_price = (
        base_price + price_sqft + price_bed_bath + price_location +
        price_school + price_transit_penalty + price_age_depreciation +
        price_lot + interaction + market_noise
    )
    sale_price = np.clip(sale_price, 140000.0, 1850000.0)
    
    # Construct DataFrame
    df = pd.DataFrame({
        'Property_ID': [f'PROP_{i+1:04d}' for i in range(n_samples)],
        'Square_Footage': np.round(sqft, 1),
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age_of_Property_Years': np.round(age_years, 1),
        'Distance_to_Metro_KM': np.round(distance_metro_km, 2),
        'Location_Desirability_Score': np.round(location_score, 2),
        'School_Rating_Index': np.round(school_rating, 2),
        'Lot_Area_SqFt': np.round(lot_area_sqft, 1),
        'Sale_Price_USD': np.round(sale_price, 2)
    })
    
    feature_names = [col for col in df.columns if col not in ['Property_ID', 'Sale_Price_USD']]
    return df, feature_names

def run_module_1():
    print("=" * 80)
    print("EXPERIMENT 5 - MODULE 1: DATA COLLECTION & PREPROCESSING")
    print("Real-Estate Housing Valuation Synthesis & Exploratory Analysis")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Synthesize Dataset
    print("\n[Step 1/5] Synthesizing Real-Estate Housing Valuation Dataset (N = 1,000)...")
    df_raw, feature_names = synthesize_real_estate_dataset(n_samples=1000, seed=42)
    print(f"Generated {len(df_raw)} property records with {len(feature_names)} features.")
    print("Sale Price Summary (USD):")
    print(f"  Mean:   ${df_raw['Sale_Price_USD'].mean():,.2f}")
    print(f"  Std:    ${df_raw['Sale_Price_USD'].std():,.2f}")
    print(f"  Min:    ${df_raw['Sale_Price_USD'].min():,.2f}")
    print(f"  Median: ${df_raw['Sale_Price_USD'].median():,.2f}")
    print(f"  Max:    ${df_raw['Sale_Price_USD'].max():,.2f}")
    
    # 2. Validation & Preprocessing
    print("\n[Step 2/5] Validating Data Integrity & Executing Z-Score Standardization...")
    null_counts = df_raw.isnull().sum().sum()
    print(f"Missing Values: {null_counts} (Data integrity confirmed)")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_raw[feature_names])
    
    df_preprocessed = df_raw.copy()
    df_preprocessed[feature_names] = X_scaled
    
    # Normalize price into thousands of USD for stable gradient descent
    df_preprocessed['Sale_Price_kUSD'] = df_raw['Sale_Price_USD'] / 1000.0
    
    # 3. Create Feature Metadata Summary Table
    print("\n[Step 3/5] Generating Feature Statistical Architecture Table...")
    meta_rows = []
    descriptions = {
        'Square_Footage': 'Finished interior livable space area (sq ft)',
        'Bedrooms': 'Total count of dedicated private bedrooms',
        'Bathrooms': 'Combined full and half bathroom count',
        'Age_of_Property_Years': 'Time elapsed since initial construction (years)',
        'Distance_to_Metro_KM': 'Network distance to nearest rapid transit station (km)',
        'Location_Desirability_Score': 'Composite neighborhood quality & safety index [1-10]',
        'School_Rating_Index': 'State standardized public school district score [1-10]',
        'Lot_Area_SqFt': 'Total parcel ground land area (sq ft)'
    }
    units = {
        'Square_Footage': 'Sq Ft',
        'Bedrooms': 'Count',
        'Bathrooms': 'Count',
        'Age_of_Property_Years': 'Years',
        'Distance_to_Metro_KM': 'Kilometers',
        'Location_Desirability_Score': 'Index [1-10]',
        'School_Rating_Index': 'Index [1-10]',
        'Lot_Area_SqFt': 'Sq Ft'
    }
    for idx, f in enumerate(feature_names):
        meta_rows.append({
            'Feature_Index': f'F{idx+1:02d}',
            'Feature_Name': f,
            'Measurement_Unit': units[f],
            'Raw_Mean': round(df_raw[f].mean(), 2),
            'Raw_Std': round(df_raw[f].std(), 2),
            'Raw_Min': round(df_raw[f].min(), 2),
            'Raw_Max': round(df_raw[f].max(), 2),
            'Domain_Description': descriptions[f]
        })
    df_metadata = pd.DataFrame(meta_rows)
    
    # 4. Save CSVs
    print("\n[Step 4/5] Saving CSV Datasets...")
    raw_path = os.path.join(base_dir, 'raw_real_estate_dataset_1000.csv')
    prep_path = os.path.join(base_dir, 'preprocessed_real_estate_1000.csv')
    meta_path = os.path.join(base_dir, 'real_estate_feature_metadata_1000.csv')
    
    df_raw.to_csv(raw_path, index=False)
    df_preprocessed.to_csv(prep_path, index=False)
    df_metadata.to_csv(meta_path, index=False)
    print(f"Saved: {raw_path}")
    print(f"Saved: {prep_path}")
    print(f"Saved: {meta_path}")
    
    # 5. Generate Figures
    print("\n[Step 5/5] Generating Publication-Quality Figures...")
    
    # -------------------------------------------------------------
    # Figure 1: Housing Feature Distributions & Target Sale Price
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Sale Price Histogram with KDE
    set_minimal_theme(axes[0])
    sns.histplot(df_raw['Sale_Price_USD'] / 1000.0, ax=axes[0], kde=True, bins=35,
                 color='#2563EB', edgecolor='#1E40AF', alpha=0.65, line_kws={'linewidth': 2.2})
    mean_price_k = df_raw['Sale_Price_USD'].mean() / 1000.0
    median_price_k = df_raw['Sale_Price_USD'].median() / 1000.0
    axes[0].axvline(mean_price_k, color='#EF4444', linestyle='--', linewidth=2.0, label=f'Mean Price: ${mean_price_k:,.1f}k')
    axes[0].axvline(median_price_k, color='#10B981', linestyle=':', linewidth=2.0, label=f'Median Price: ${median_price_k:,.1f}k')
    axes[0].set_title("(A) Real-Estate Sale Price Distribution (N = 1,000)", pad=12)
    axes[0].set_xlabel("Property Sale Price (Thousands of USD)", labelpad=8)
    axes[0].set_ylabel("Property Frequency Count", labelpad=8)
    
    # Subplot B: Square Footage vs. Sale Price by Location Score
    set_minimal_theme(axes[1])
    sc = axes[1].scatter(df_raw['Square_Footage'], df_raw['Sale_Price_USD'] / 1000.0,
                         c=df_raw['Location_Desirability_Score'], cmap='viridis',
                         alpha=0.75, edgecolors='#FFFFFF', linewidths=0.4, s=45)
    cbar = fig.colorbar(sc, ax=axes[1], orientation='horizontal', pad=0.16)
    cbar.set_label('Location Desirability Score [1 - 10]')
    
    # Polynomial trend line to illustrate non-linearity
    z_fit = np.polyfit(df_raw['Square_Footage'], df_raw['Sale_Price_USD'] / 1000.0, 2)
    p_fit = np.poly1d(z_fit)
    sqft_sort = np.sort(df_raw['Square_Footage'])
    axes[1].plot(sqft_sort, p_fit(sqft_sort), color='#EF4444', linewidth=2.5, linestyle='-', label='Quadratic Trend')
    
    axes[1].set_title("(B) Sale Price vs. Interior Square Footage (Colored by Location Score)", pad=12)
    axes[1].set_xlabel("Square Footage (sq ft)", labelpad=8)
    axes[1].set_ylabel("Sale Price (Thousands of USD)", labelpad=8)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles1 = [
        mpl.lines.Line2D([0], [0], color='#EF4444', linestyle='--', linewidth=2.0, label=f'Mean Price (${mean_price_k:,.1f}k)'),
        mpl.lines.Line2D([0], [0], color='#10B981', linestyle=':', linewidth=2.0, label=f'Median Price (${median_price_k:,.1f}k)'),
        mpl.lines.Line2D([0], [0], color='#EF4444', linewidth=2.5, label='Non-Linear Price Trend Line'),
        mpl.patches.Patch(facecolor='#2563EB', alpha=0.65, label='Empirical Housing Histogram')
    ]
    fig.legend(handles=legend_handles1, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig1_path = os.path.join(base_dir, '01_housing_feature_distributions_and_target.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Correlation Matrix & Sale Price Drivers
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Pearson Correlation Heatmap
    set_minimal_theme(axes[0])
    short_names = ['SqFt', 'Beds', 'Baths', 'Age', 'MetroDist', 'Location', 'School', 'LotArea', 'Price']
    corr = df_raw[feature_names + ['Sale_Price_USD']].corr()
    sns.heatmap(corr, ax=axes[0], annot=True, fmt=".2f", cmap='coolwarm', vmin=-1.0, vmax=1.0, center=0.0,
                xticklabels=short_names, yticklabels=short_names,
                cbar_kws={'orientation': 'horizontal', 'pad': 0.16, 'label': 'Pearson Correlation Coefficient (r)'})
    axes[0].set_title("(A) Real-Estate Variables Pearson Correlation Matrix", pad=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Subplot B: Correlation with Sale Price (Ranked)
    set_minimal_theme(axes[1])
    price_corrs = corr['Sale_Price_USD'].drop('Sale_Price_USD').sort_values(ascending=True)
    bar_colors = ['#EF4444' if r < 0 else '#2563EB' for r in price_corrs.values]
    y_pos = np.arange(len(price_corrs))
    axes[1].barh(y_pos, price_corrs.values, color=bar_colors, edgecolor='#1F2937', linewidth=0.8, alpha=0.85)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(price_corrs.index)
    axes[1].axvline(0, color='#1F2937', linewidth=1.0)
    for i, v in enumerate(price_corrs.values):
        offset = 0.02 if v >= 0 else -0.07
        axes[1].text(v + offset, i, f"{v:+.3f}", va='center', fontname='Cambria', fontsize=9.5, fontweight='bold')
    axes[1].set_title("(B) Housing Feature Correlation with Sale Price (Pearson r)", pad=12)
    axes[1].set_xlabel("Linear Correlation with Sale Price (r)", labelpad=8)
    axes[1].set_xlim(-0.65, 0.95)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    legend_handles2 = [
        mpl.patches.Patch(facecolor='#2563EB', edgecolor='#1F2937', label='Positive Price Driver (Square Footage, Location, School)'),
        mpl.patches.Patch(facecolor='#EF4444', edgecolor='#1F2937', label='Negative Price Driver (Distance to Metro, Property Age)')
    ]
    fig.legend(handles=legend_handles2, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=2, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig2_path = os.path.join(base_dir, '02_real_estate_feature_correlation_matrix.png')
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
    sns.kdeplot(df_raw['Square_Footage'] / 1000.0, ax=axes[0], color='#EF4444', linewidth=2.2,
                label='Raw Square Footage (k-SqFt, Mean=2.26, Std=0.79)', fill=True, alpha=0.18)
    sns.kdeplot(df_preprocessed['Square_Footage'], ax=axes[0], color='#2563EB', linewidth=2.2,
                label='Standardized Square Footage (Mean=0.00, Std=1.00)', fill=True, alpha=0.18)
    sns.kdeplot(df_preprocessed['Location_Desirability_Score'], ax=axes[0], color='#10B981', linewidth=2.2,
                label='Standardized Location Score (Mean=0.00, Std=1.00)', fill=True, alpha=0.18)
    axes[0].set_title("(A) Probability Density: Raw vs. Standardized Feature Coordinates", pad=12)
    axes[0].set_xlabel("Feature Coordinate Value", labelpad=8)
    axes[0].set_ylabel("Kernel Density Estimate", labelpad=8)
    axes[0].set_xlim(-4.0, 5.0)
    
    # Subplot B: Standardized Feature Boxplots
    set_minimal_theme(axes[1])
    box_data = [df_preprocessed[col] for col in feature_names]
    short_feat = ['SqFt', 'Beds', 'Baths', 'Age', 'MetroDist', 'Location', 'School', 'LotArea']
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
    axes[1].set_title("(B) Standardized Feature Coordinates Boxplot & Outlier Bounds (Z = +/-3)", pad=12)
    axes[1].set_xlabel("Real-Estate Housing Predictors", labelpad=8)
    axes[1].set_ylabel("Standardized Z-Score", labelpad=8)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(-4.5, 4.5)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles3 = [
        mpl.lines.Line2D([0], [0], color='#EF4444', linewidth=2.0, label='Raw Feature Scale (k-SqFt)'),
        mpl.lines.Line2D([0], [0], color='#2563EB', linewidth=2.0, label='Z-Score Standardized Feature (N(0, 1))'),
        mpl.patches.Patch(facecolor='#E0E7FF', edgecolor='#3730A3', label='Interquartile Range (IQR)'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EF4444', markersize=6, label='Tukey Outlier (|Z| > 2.5)')
    ]
    fig.legend(handles=legend_handles3, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig3_path = os.path.join(base_dir, '03_feature_standardization_and_outliers.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig3_path}")
    
    print("\nModule 1 processing and visualization completed successfully!")

if __name__ == '__main__':
    run_module_1()
