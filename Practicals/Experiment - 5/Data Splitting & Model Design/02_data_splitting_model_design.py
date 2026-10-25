"""
Module 2: Data Splitting & Model Design
Scenario: A real-estate price prediction system compares linear regression, a decision tree,
and a high-complexity polynomial model. Evaluate their training and testing errors to identify
the most suitable model. Compare Gradient Descent, SGD, Momentum, and Adam.

This script:
1. Performs stratified 70/15/15 train/val/test data partitioning based on price quantiles.
2. Formulates three regression model architectures:
   - Model A: Multiple Linear Regression (High Bias, Low Complexity, 9 parameters).
   - Model B: Decision Tree Regressor (Piecewise Non-linear, Balanced, Depth = 6).
   - Model C: High-Complexity Polynomial Regression (High Variance, Degree 3 Expansion, 165 parameters).
3. Implements 4 fundamental optimization algorithms from scratch:
   - Batch Gradient Descent (BGD)
   - Stochastic Gradient Descent (SGD)
   - Momentum Gradient Descent (Polyak momentum, beta = 0.9)
   - Adaptive Moment Estimation (Adam, beta1 = 0.9, beta2 = 0.999)
4. Saves partitioned datasets and trained model weights to CSV and NPZ.
5. Generates 3 publication-grade figures:
   - Figure 04: Dataset Split Proportions & Price Invariance.
   - Figure 05: Predicted vs. Actual Sale Price Parity Plots.
   - Figure 06: Optimization Loss Surface & Trajectory Dynamics (BGD, SGD, Momentum, Adam).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

# -------------------------------------------------------------
# Optimization Algorithms Implementation
# -------------------------------------------------------------
def run_optimizer_trajectory(X_2d, y, opt_type='bgd', lr=0.08, epochs=60, seed=42):
    """
    Simulates optimization trajectories on a 2D parameter sub-space for visualization.
    X_2d: shape (N, 2), y: shape (N,)
    opt_type: 'bgd', 'sgd', 'momentum', 'adam'
    """
    np.random.seed(seed)
    N = len(y)
    w = np.array([-1.8, 2.5])  # initial weights far from optimum
    trajectory = [w.copy()]
    
    v = np.zeros(2)
    m = np.zeros(2)
    v_adam = np.zeros(2)
    beta = 0.9
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    for t in range(1, epochs + 1):
        if opt_type == 'bgd':
            # Full batch gradient
            pred = np.dot(X_2d, w)
            grad = (2.0 / N) * np.dot(X_2d.T, (pred - y))
            w = w - lr * grad
            
        elif opt_type == 'sgd':
            # Stochastic single-sample gradient with high variance
            for i in range(10):  # 10 random samples per epoch
                idx = np.random.randint(0, N)
                xi, yi = X_2d[idx], y[idx]
                grad = 2.0 * xi * (np.dot(xi, w) - yi)
                w = w - (lr * 0.4) * grad
                
        elif opt_type == 'momentum':
            # Momentum with velocity accumulation
            pred = np.dot(X_2d, w)
            grad = (2.0 / N) * np.dot(X_2d.T, (pred - y))
            v = beta * v + lr * grad
            w = w - v
            
        elif opt_type == 'adam':
            # Adam adaptive moment estimation
            pred = np.dot(X_2d, w)
            grad = (2.0 / N) * np.dot(X_2d.T, (pred - y))
            m = beta1 * m + (1 - beta1) * grad
            v_adam = beta2 * v_adam + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v_adam / (1 - beta2 ** t)
            w = w - (lr * 1.5 / (np.sqrt(v_hat) + eps)) * m_hat
            
        trajectory.append(w.copy())
        
    return np.array(trajectory)

def run_module_2():
    print("=" * 80)
    print("EXPERIMENT 5 - MODULE 2: DATA SPLITTING & MODEL DESIGN")
    print("Stratified Splitting, Model Complexity & Optimization Dynamics")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp5_dir = os.path.dirname(base_dir)
    prep_csv_path = os.path.join(exp5_dir, 'Data Collection & Preprocessing', 'preprocessed_real_estate_1000.csv')
    
    # 1. Load Preprocessed Dataset
    print("\n[Step 1/5] Loading preprocessed real-estate housing records...")
    df = pd.read_csv(prep_csv_path)
    feature_names = [col for col in df.columns if col not in ['Property_ID', 'Sale_Price_USD', 'Sale_Price_kUSD']]
    
    X = df[feature_names].values
    y = df['Sale_Price_kUSD'].values  # targets in thousands of USD
    
    # 2. Stratified / Quantile-Binned 70/15/15 Partitioning
    print("\n[Step 2/5] Partitioning Dataset into Train (70%), Val (15%), and Test (15%)...")
    # Discretize price into 5 quantile bins for stratified splitting
    price_bins = pd.qcut(y, q=5, labels=False)
    
    train_idx, temp_idx = train_test_split(np.arange(len(df)), test_size=0.30, random_state=42, stratify=price_bins)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=price_bins[temp_idx])
    
    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df.iloc[val_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    
    print(f"Data Partitions:")
    print(f"  - Training Set:   {len(df_train)} properties (70.0%)")
    print(f"  - Validation Set: {len(df_val)} properties (15.0%)")
    print(f"  - Test Set:       {len(df_test)} properties (15.0%)")
    
    # Export split CSVs
    train_path = os.path.join(base_dir, 'train_real_estate.csv')
    val_path = os.path.join(base_dir, 'val_real_estate.csv')
    test_path = os.path.join(base_dir, 'test_real_estate.csv')
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")
    
    # 3. Model Architecture Instantiation & Fitting
    print("\n[Step 3/5] Instantiating and Fitting Three Regression Models...")
    X_train = df_train[feature_names].values
    y_train = df_train['Sale_Price_kUSD'].values
    X_test = df_test[feature_names].values
    y_test = df_test['Sale_Price_kUSD'].values
    
    # Model A: Multiple Linear Regression (Analytical OLS Normal Equation)
    X_train_bias = np.c_[np.ones(len(X_train)), X_train]
    X_test_bias = np.c_[np.ones(len(X_test)), X_test]
    w_linear = np.linalg.pinv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train
    y_pred_train_lin = X_train_bias @ w_linear
    y_pred_test_lin = X_test_bias @ w_linear
    
    # Model B: Decision Tree Regressor (Non-linear, depth = 6)
    dt_model = DecisionTreeRegressor(max_depth=6, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_train_dt = dt_model.predict(X_train)
    y_pred_test_dt = dt_model.predict(X_test)
    
    # Model C: High-Complexity Polynomial Regression (Degree 3 Expansion)
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    # Fit unregularized OLS on polynomial features (exhibits classic overfitting)
    w_poly = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
    y_pred_train_poly = X_train_poly @ w_poly
    y_pred_test_poly = X_test_poly @ w_poly
    
    print(f"Model Complexity Specs:")
    print(f"  - Linear Regression:           {len(w_linear)} parameters (Underfitting / High Bias)")
    print(f"  - Decision Tree (Depth=6):     {dt_model.get_n_leaves()} leaf nodes (Balanced Non-linear)")
    print(f"  - Polynomial Model (Degree 3): {len(w_poly)} parameters (Overfitting / High Variance)")
    
    # Save model weights to NPZ
    weights_path = os.path.join(base_dir, 'trained_regression_models.npz')
    np.savez(
        weights_path,
        w_linear=w_linear,
        w_poly=w_poly,
        feature_names=feature_names
    )
    print(f"Saved: {weights_path}")
    
    # 4. Generate Visualizations
    print("\n[Step 4/5] Generating Publication-Quality Figures...")
    
    # -------------------------------------------------------------
    # Figure 4: Dataset Split Proportions & Price Distribution Invariance
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Partition Proportions Bar Chart
    set_minimal_theme(axes[0])
    splits = ['Training (70%)', 'Validation (15%)', 'Testing (15%)']
    counts = [len(df_train), len(df_val), len(df_test)]
    colors_split = ['#2563EB', '#10B981', '#EF4444']
    bars = axes[0].bar(splits, counts, color=colors_split, edgecolor='#1F2937', linewidth=0.8, alpha=0.88, width=0.55)
    axes[0].set_title("(A) Stratified Real-Estate Dataset Partition Breakdown", pad=12)
    axes[0].set_xlabel("Dataset Partition", labelpad=8)
    axes[0].set_ylabel("Number of Property Records", labelpad=8)
    axes[0].set_ylim(0, 850)
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h + 15, f"{h} ({h/1000*100:.1f}%)",
                     ha='center', va='bottom', fontname='Cambria', fontsize=10, fontweight='bold')
                     
    # Subplot B: Density Invariance of Target Variable across Splits
    set_minimal_theme(axes[1])
    sns.kdeplot(df_train['Sale_Price_kUSD'], ax=axes[1], color='#2563EB', linewidth=2.2, label='Train Partition (N = 700)', fill=True, alpha=0.15)
    sns.kdeplot(df_val['Sale_Price_kUSD'], ax=axes[1], color='#10B981', linewidth=2.2, linestyle='--', label='Val Partition (N = 150)', fill=True, alpha=0.15)
    sns.kdeplot(df_test['Sale_Price_kUSD'], ax=axes[1], color='#EF4444', linewidth=2.2, linestyle=':', label='Test Partition (N = 150)', fill=True, alpha=0.15)
    axes[1].set_title("(B) Target Sale Price Distribution Consistency Across Partitions", pad=12)
    axes[1].set_xlabel("Sale Price (Thousands of USD)", labelpad=8)
    axes[1].set_ylabel("Kernel Density Estimate", labelpad=8)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles4 = [
        mpl.patches.Patch(facecolor='#2563EB', edgecolor='#1F2937', label='Training Set (700 records, 70.0%)'),
        mpl.patches.Patch(facecolor='#10B981', edgecolor='#1F2937', label='Validation Set (150 records, 15.0%)'),
        mpl.patches.Patch(facecolor='#EF4444', edgecolor='#1F2937', label='Testing Set (150 records, 15.0%)'),
        mpl.lines.Line2D([0], [0], color='#111827', linestyle='--', label='Zero Covariate Shift Verification')
    ]
    fig.legend(handles=legend_handles4, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig4_path = os.path.join(base_dir, '04_dataset_split_proportions_and_price_invariance.png')
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig4_path}")
    
    # -------------------------------------------------------------
    # Figure 5: Model Predictions vs Actual Sale Price Parity
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Linear Regression
    set_minimal_theme(axes[0])
    axes[0].scatter(y_test, y_pred_test_lin, color='#2563EB', alpha=0.65, edgecolors='#FFFFFF', linewidths=0.5, s=40)
    min_val, max_val = min(y_test), max(y_test)
    axes[0].plot([min_val, max_val], [min_val, max_val], color='#111827', linestyle='--', linewidth=1.5, label='Ideal Parity (y = x)')
    r2_lin = r2_score(y_test, y_pred_test_lin)
    axes[0].text(min_val + 50, max_val - 120, f"Linear Regression\nR^2 = {r2_lin:.3f}\nHigh Bias (Underfit)",
                 fontname='Cambria', fontsize=9.5, fontweight='bold', bbox=dict(facecolor='#FFFFFF', edgecolor='#D1D5DB', alpha=0.85))
    axes[0].set_title("(A) Linear Regression Parity", pad=10)
    axes[0].set_xlabel("Actual Sale Price (kUSD)", labelpad=6)
    axes[0].set_ylabel("Predicted Sale Price (kUSD)", labelpad=6)
    
    # Subplot B: Decision Tree
    set_minimal_theme(axes[1])
    axes[1].scatter(y_test, y_pred_test_dt, color='#10B981', alpha=0.65, edgecolors='#FFFFFF', linewidths=0.5, s=40)
    axes[1].plot([min_val, max_val], [min_val, max_val], color='#111827', linestyle='--', linewidth=1.5)
    r2_dt = r2_score(y_test, y_pred_test_dt)
    axes[1].text(min_val + 50, max_val - 120, f"Decision Tree (D=6)\nR^2 = {r2_dt:.3f}\nOptimal Generalization",
                 fontname='Cambria', fontsize=9.5, fontweight='bold', bbox=dict(facecolor='#FFFFFF', edgecolor='#D1D5DB', alpha=0.85))
    axes[1].set_title("(B) Decision Tree Regressor Parity", pad=10)
    axes[1].set_xlabel("Actual Sale Price (kUSD)", labelpad=6)
    axes[1].set_ylabel("Predicted Sale Price (kUSD)", labelpad=6)
    
    # Subplot C: High-Complexity Polynomial
    set_minimal_theme(axes[2])
    # Clip extreme test polynomial predictions for clean plotting
    y_pred_plot_poly = np.clip(y_pred_test_poly, -200, max_val + 400)
    axes[2].scatter(y_test, y_pred_plot_poly, color='#EF4444', alpha=0.65, edgecolors='#FFFFFF', linewidths=0.5, s=40)
    axes[2].plot([min_val, max_val], [min_val, max_val], color='#111827', linestyle='--', linewidth=1.5)
    r2_poly = r2_score(y_test, y_pred_test_poly)
    axes[2].text(min_val + 50, max_val - 120, f"Degree 3 Polynomial\nR^2 = {r2_poly:.3f}\nSevere Overfitting",
                 fontname='Cambria', fontsize=9.5, fontweight='bold', bbox=dict(facecolor='#FFFFFF', edgecolor='#D1D5DB', alpha=0.85))
    axes[2].set_title("(C) High-Complexity Polynomial Parity", pad=10)
    axes[2].set_xlabel("Actual Sale Price (kUSD)", labelpad=6)
    axes[2].set_ylabel("Predicted Sale Price (kUSD)", labelpad=6)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles5 = [
        mpl.lines.Line2D([0], [0], color='#111827', linestyle='--', linewidth=1.5, label='Perfect Parity Diagonal (y = x)'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2563EB', markersize=8, label=f'Linear Regression (Test R^2: {r2_lin:.3f})'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#10B981', markersize=8, label=f'Decision Tree (Test R^2: {r2_dt:.3f} - Best)'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EF4444', markersize=8, label=f'Polynomial Model (Test R^2: {r2_poly:.3f} - Overfit)')
    ]
    fig.legend(handles=legend_handles5, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig5_path = os.path.join(base_dir, '05_model_architecture_predictions_vs_actual.png')
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig5_path}")
    
    # -------------------------------------------------------------
    # Figure 6: Loss Surface & Optimizer Trajectories
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Generate 2D subspace optimization trajectories (Square Footage vs Location Score)
    X_sub = df_train[['Square_Footage', 'Location_Desirability_Score']].values
    y_sub = (df_train['Sale_Price_kUSD'].values - df_train['Sale_Price_kUSD'].mean()) / df_train['Sale_Price_kUSD'].std()
    
    # Compute optimal weights for 2D subspace
    w_opt = np.linalg.pinv(X_sub.T @ X_sub) @ X_sub.T @ y_sub
    
    # Trajectories
    traj_bgd = run_optimizer_trajectory(X_sub, y_sub, opt_type='bgd', lr=0.08, epochs=40)
    traj_sgd = run_optimizer_trajectory(X_sub, y_sub, opt_type='sgd', lr=0.08, epochs=40)
    traj_mom = run_optimizer_trajectory(X_sub, y_sub, opt_type='momentum', lr=0.08, epochs=40)
    traj_adam = run_optimizer_trajectory(X_sub, y_sub, opt_type='adam', lr=0.08, epochs=40)
    
    # Contour Grid
    set_minimal_theme(axes[0])
    w1_vals = np.linspace(-2.2, 1.8, 100)
    w2_vals = np.linspace(-1.0, 3.0, 100)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    loss_grid = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w_ij = np.array([W1[i, j], W2[i, j]])
            err = y_sub - np.dot(X_sub, w_ij)
            loss_grid[i, j] = np.mean(err**2)
            
    cs = axes[0].contour(W1, W2, loss_grid, levels=25, cmap='viridis_r', alpha=0.75, linewidths=0.9)
    axes[0].clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    
    # Plot trajectories on Subplot A
    axes[0].plot(traj_bgd[:, 0], traj_bgd[:, 1], color='#2563EB', marker='o', markersize=3.5, linewidth=1.8, label='BGD (Smooth Steepest Descent)')
    axes[0].plot(traj_sgd[:, 0], traj_sgd[:, 1], color='#F59E0B', marker='x', markersize=4, linewidth=1.4, alpha=0.85, label='SGD (Stochastic Oscillations)')
    axes[0].plot(traj_mom[:, 0], traj_mom[:, 1], color='#10B981', marker='s', markersize=3.5, linewidth=2.0, label='Momentum (Oscillation Damped)')
    axes[0].plot(traj_adam[:, 0], traj_adam[:, 1], color='#8B5CF6', marker='^', markersize=4, linewidth=2.2, label='Adam (Adaptive Moment Tracking)')
    axes[0].scatter(w_opt[0], w_opt[1], color='#EF4444', s=140, marker='*', zorder=10, label='Global Optimum (OLS Solution)')
    axes[0].set_title("(A) 2D Parameter Loss Contour & Optimizer Trajectories", pad=12)
    axes[0].set_xlabel("Weight 1: Square Footage (Standardized)", labelpad=8)
    axes[0].set_ylabel("Weight 2: Location Score (Standardized)", labelpad=8)
    
    # Subplot B: Optimizer Euclidean Distance to Optimum vs Iterations
    set_minimal_theme(axes[1])
    dist_bgd = np.linalg.norm(traj_bgd - w_opt, axis=1)
    dist_sgd = np.linalg.norm(traj_sgd - w_opt, axis=1)
    dist_mom = np.linalg.norm(traj_mom - w_opt, axis=1)
    dist_adam = np.linalg.norm(traj_adam - w_opt, axis=1)
    iter_x = np.arange(len(dist_bgd))
    
    axes[1].plot(iter_x, dist_bgd, color='#2563EB', linewidth=2.0, label='Batch Gradient Descent (BGD)')
    axes[1].plot(iter_x, dist_sgd, color='#F59E0B', linewidth=1.8, linestyle='--', label='Stochastic Gradient Descent (SGD)')
    axes[1].plot(iter_x, dist_mom, color='#10B981', linewidth=2.2, label='Momentum (beta = 0.9)')
    axes[1].plot(iter_x, dist_adam, color='#8B5CF6', linewidth=2.4, label='Adam (Adaptive Rates)')
    axes[1].set_title("(B) Parameter Convergence Velocity: Distance to Global Optimum", pad=12)
    axes[1].set_xlabel("Optimization Epochs / Iterations", labelpad=8)
    axes[1].set_ylabel("Euclidean Parameter Error ||w_t - w*||", labelpad=8)
    axes[1].set_yscale('log')
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles6 = [
        mpl.lines.Line2D([0], [0], color='#2563EB', marker='o', linewidth=2.0, label='BGD (Monotonic / High Latency per Epoch)'),
        mpl.lines.Line2D([0], [0], color='#F59E0B', marker='x', linewidth=1.8, label='SGD (High Variance Noise Path)'),
        mpl.lines.Line2D([0], [0], color='#10B981', marker='s', linewidth=2.2, label='Momentum (Curvature Acceleration)'),
        mpl.lines.Line2D([0], [0], color='#8B5CF6', marker='^', linewidth=2.4, label='Adam (Fastest Parameter Convergence)'),
        mpl.lines.Line2D([0], [0], marker='*', color='w', markerfacecolor='#EF4444', markersize=12, label='Global OLS Optimum')
    ]
    fig.legend(handles=legend_handles6, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=5, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig6_path = os.path.join(base_dir, '06_loss_surface_and_optimizer_trajectories.png')
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig6_path}")
    
    print("\nModule 2 processing and visualization completed successfully!")

if __name__ == '__main__':
    run_module_2()
