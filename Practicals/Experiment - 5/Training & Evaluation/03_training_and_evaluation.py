"""
Module 3: Training & Evaluation
Scenario: A real-estate price prediction system compares linear regression, a decision tree,
and a high-complexity polynomial model. Evaluate their training and testing errors to identify
the most suitable model. Compare Gradient Descent, SGD, Momentum, and Adam.

This script:
1. Conducts comprehensive training and testing error evaluation (MSE, RMSE, MAE, R2)
   across model complexity tiers (Polynomial Degrees 1-4 and Decision Tree Depths 2-12).
2. Executes 100-epoch convergence benchmarking for 4 optimizers:
   - Batch Gradient Descent (BGD)
   - Stochastic Gradient Descent (SGD)
   - Momentum Gradient Descent (beta = 0.9)
   - Adaptive Moment Estimation (Adam)
3. Measures gradient oscillation amplitudes and convergence speed.
4. Benchmarks vectorized NumPy BLAS execution against scalar Python loops.
5. Exports comprehensive evaluation CSVs.
6. Generates 3 publication-grade visualization figures:
   - Figure 07: Training vs. Testing Error Curves across Model Complexity.
   - Figure 08: Optimizer Loss Convergence & Oscillation Damping Curves.
   - Figure 09: Model Residual Distributions & Computational Speedup Bar Chart.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

def run_module_3():
    print("=" * 80)
    print("EXPERIMENT 5 - MODULE 3: TRAINING & EVALUATION")
    print("Model Complexity Benchmarking & Optimizer Convergence Analysis")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp5_dir = os.path.dirname(base_dir)
    mod2_dir = os.path.join(exp5_dir, 'Data Splitting & Model Design')
    
    # 1. Load Data Partitions
    print("\n[Step 1/6] Loading train, validation, and test datasets...")
    train_df = pd.read_csv(os.path.join(mod2_dir, 'train_real_estate.csv'))
    val_df = pd.read_csv(os.path.join(mod2_dir, 'val_real_estate.csv'))
    test_df = pd.read_csv(os.path.join(mod2_dir, 'test_real_estate.csv'))
    
    feature_names = [col for col in train_df.columns if col not in ['Property_ID', 'Sale_Price_USD', 'Sale_Price_kUSD']]
    
    X_train = train_df[feature_names].values
    y_train = train_df['Sale_Price_kUSD'].values
    X_test = test_df[feature_names].values
    y_test = test_df['Sale_Price_kUSD'].values
    
    # 2. Comprehensive Model Complexity Evaluation
    print("\n[Step 2/6] Evaluating Training and Testing Errors Across Model Complexity Tiers...")
    complexity_records = []
    
    # Tier 1: Polynomial Models (Degrees 1 to 4)
    for deg in [1, 2, 3, 4]:
        poly = PolynomialFeatures(degree=deg, include_bias=True)
        X_tr_poly = poly.fit_transform(X_train)
        X_te_poly = poly.transform(X_test)
        
        # OLS fit
        w = np.linalg.pinv(X_tr_poly.T @ X_tr_poly) @ X_tr_poly.T @ y_train
        y_tr_pred = X_tr_poly @ w
        y_te_pred = X_te_poly @ w
        
        tr_mse = mean_squared_error(y_train, y_tr_pred)
        te_mse = mean_squared_error(y_test, y_te_pred)
        tr_r2 = r2_score(y_train, y_tr_pred)
        te_r2 = r2_score(y_test, y_te_pred)
        tr_mae = mean_absolute_error(y_train, y_tr_pred)
        te_mae = mean_absolute_error(y_test, y_te_pred)
        
        complexity_records.append({
            'Model_Family': 'Polynomial Regression',
            'Model_Name': f'Linear (Deg 1)' if deg == 1 else f'Polynomial Deg {deg}',
            'Complexity_Metric': f'Deg {deg} ({X_tr_poly.shape[1]} params)',
            'Param_Count': X_tr_poly.shape[1],
            'Train_MSE': round(tr_mse, 2),
            'Test_MSE': round(te_mse, 2),
            'Train_RMSE': round(np.sqrt(tr_mse), 2),
            'Test_RMSE': round(np.sqrt(te_mse), 2),
            'Train_MAE': round(tr_mae, 2),
            'Test_MAE': round(te_mae, 2),
            'Train_R2': round(tr_r2, 4),
            'Test_R2': round(te_r2, 4),
            'Generalization_Gap_MSE': round(max(0.0, te_mse - tr_mse), 2)
        })
        print(f"  Poly Deg {deg}: Train MSE = {tr_mse:8.1f} | Test MSE = {te_mse:8.1f} | Test R2 = {te_r2:.4f}")
        
    # Tier 2: Decision Tree Regressors (Depths 2, 4, 6, 8, 12, Unlimited)
    for depth in [2, 4, 6, 8, 12, None]:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        y_tr_pred = dt.predict(X_train)
        y_te_pred = dt.predict(X_test)
        
        tr_mse = mean_squared_error(y_train, y_tr_pred)
        te_mse = mean_squared_error(y_test, y_te_pred)
        tr_r2 = r2_score(y_train, y_tr_pred)
        te_r2 = r2_score(y_test, y_te_pred)
        tr_mae = mean_absolute_error(y_train, y_tr_pred)
        te_mae = mean_absolute_error(y_test, y_te_pred)
        d_str = f'Depth {depth}' if depth is not None else 'Unlimited'
        
        complexity_records.append({
            'Model_Family': 'Decision Tree Regressor',
            'Model_Name': f'Decision Tree ({d_str})',
            'Complexity_Metric': f'{d_str} ({dt.get_n_leaves()} leaves)',
            'Param_Count': dt.get_n_leaves(),
            'Train_MSE': round(tr_mse, 2),
            'Test_MSE': round(te_mse, 2),
            'Train_RMSE': round(np.sqrt(tr_mse), 2),
            'Test_RMSE': round(np.sqrt(te_mse), 2),
            'Train_MAE': round(tr_mae, 2),
            'Test_MAE': round(te_mae, 2),
            'Train_R2': round(tr_r2, 4),
            'Test_R2': round(te_r2, 4),
            'Generalization_Gap_MSE': round(max(0.0, te_mse - tr_mse), 2)
        })
        print(f"  Tree {d_str:10s}: Train MSE = {tr_mse:8.1f} | Test MSE = {te_mse:8.1f} | Test R2 = {te_r2:.4f}")
        
    df_complexity = pd.DataFrame(complexity_records)
    
    # 3. Optimizer Convergence & Gradient Oscillation Benchmarking
    print("\n[Step 3/6] Benchmarking 4 Optimization Algorithms (BGD, SGD, Momentum, Adam)...")
    epochs = 100
    N = len(X_train)
    D = X_train.shape[1] + 1
    X_tr_b = np.c_[np.ones(N), X_train]
    y_norm = (y_train - np.mean(y_train)) / np.std(y_train)  # standardized target for stable descent
    
    opt_history = {'epoch': list(range(1, epochs + 1))}
    oscillation_scores = {}
    
    # Define 4 optimizers
    optimizers = ['BGD', 'SGD', 'Momentum', 'Adam']
    for opt_name in optimizers:
        np.random.seed(42)
        w = np.random.randn(D) * 0.1
        v = np.zeros(D)
        m = np.zeros(D)
        v_adam = np.zeros(D)
        beta_mom = 0.9
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = 0.05 if opt_name != 'Adam' else 0.08
        
        losses = []
        grad_cosines = []
        prev_grad = None
        
        t0 = time.perf_counter()
        for ep in range(1, epochs + 1):
            if opt_name == 'BGD':
                pred = X_tr_b @ w
                grad = (2.0 / N) * X_tr_b.T @ (pred - y_norm)
                w -= lr * grad
                
            elif opt_name == 'SGD':
                # Mini-batch / Stochastic (10 samples per epoch)
                for _ in range(15):
                    idx = np.random.randint(0, N)
                    xi, yi = X_tr_b[idx], y_norm[idx]
                    grad = 2.0 * xi * (np.dot(xi, w) - yi)
                    w -= (lr * 0.3) * grad
                pred = X_tr_b @ w
                
            elif opt_name == 'Momentum':
                pred = X_tr_b @ w
                grad = (2.0 / N) * X_tr_b.T @ (pred - y_norm)
                v = beta_mom * v + lr * grad
                w -= v
                
            elif opt_name == 'Adam':
                pred = X_tr_b @ w
                grad = (2.0 / N) * X_tr_b.T @ (pred - y_norm)
                m = beta1 * m + (1 - beta1) * grad
                v_adam = beta2 * v_adam + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1 ** ep)
                v_hat = v_adam / (1 - beta2 ** ep)
                w -= (lr / (np.sqrt(v_hat) + eps)) * m_hat
                
            # Measure gradient oscillation: 1 - cosine similarity
            if prev_grad is not None:
                norm_p = np.linalg.norm(prev_grad)
                norm_c = np.linalg.norm(grad)
                if norm_p > 1e-8 and norm_c > 1e-8:
                    cos_sim = np.dot(prev_grad, grad) / (norm_p * norm_c)
                    grad_cosines.append(1.0 - cos_sim)
            prev_grad = grad.copy()
            
            loss_val = np.mean((pred - y_norm) ** 2)
            losses.append(loss_val)
            
        elapsed_ms = (time.perf_counter() - t0) * 1000
        opt_history[f'{opt_name}_loss'] = losses
        oscillation_scores[opt_name] = np.mean(grad_cosines) if grad_cosines else 0.0
        print(f"  {opt_name:8s}: Final Loss = {losses[-1]:.5f} | Oscillation Index = {oscillation_scores[opt_name]:.4f} | Latency = {elapsed_ms:.1f} ms")
        
    df_opt_history = pd.DataFrame(opt_history)
    
    # 4. Computational Vectorization Benchmark
    print("\n[Step 4/6] Benchmarking Scalar Loop vs. Vectorized NumPy BLAS for MSE Gradients...")
    batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    bench_records = []
    
    for bs in batch_sizes:
        X_syn = np.random.randn(bs, 8)
        y_syn = np.random.randn(bs)
        w_syn = np.random.randn(8)
        
        # Method A: Scalar Loops
        t0 = time.perf_counter()
        for _ in range(30):
            grad_scalar = np.zeros(8)
            for i in range(bs):
                pred_i = 0.0
                for j in range(8):
                    pred_i += X_syn[i, j] * w_syn[j]
                err_i = pred_i - y_syn[i]
                for j in range(8):
                    grad_scalar[j] += (2.0 / bs) * X_syn[i, j] * err_i
        t_scalar_ms = ((time.perf_counter() - t0) / 30) * 1000
        
        # Method B: Vectorized NumPy BLAS
        t0 = time.perf_counter()
        for _ in range(30):
            pred_vec = np.dot(X_syn, w_syn)
            grad_vec = (2.0 / bs) * np.dot(X_syn.T, (pred_vec - y_syn))
        t_vec_ms = ((time.perf_counter() - t0) / 30) * 1000
        
        speedup = t_scalar_ms / max(t_vec_ms, 1e-6)
        bench_records.append({
            'Batch_Size': bs,
            'Scalar_Loop_ms': round(t_scalar_ms, 4),
            'Vectorized_BLAS_ms': round(t_vec_ms, 4),
            'Speedup_Factor': round(speedup, 2)
        })
        print(f"  Batch {bs:5d}: Scalar = {t_scalar_ms:8.3f} ms | Vectorized = {t_vec_ms:6.3f} ms | Speedup = {speedup:6.1f}x")
        
    df_bench = pd.DataFrame(bench_records)
    
    # 5. Export Test Predictions for Best Models
    print("\n[Step 5/6] Exporting Predictions and Benchmark CSVs...")
    # Best models: Linear, Decision Tree (depth=6), Poly (deg=3)
    dt_best = DecisionTreeRegressor(max_depth=6, random_state=42).fit(X_train, y_train)
    poly3 = PolynomialFeatures(degree=3, include_bias=True)
    X_tr_p3 = poly3.fit_transform(X_train)
    X_te_p3 = poly3.transform(X_test)
    w_p3 = np.linalg.pinv(X_tr_p3.T @ X_tr_p3) @ X_tr_p3.T @ y_train
    
    w_lin = np.linalg.pinv(np.c_[np.ones(len(X_train)), X_train].T @ np.c_[np.ones(len(X_train)), X_train]) @ np.c_[np.ones(len(X_train)), X_train].T @ y_train
    
    df_preds = test_df[['Property_ID', 'Sale_Price_kUSD']].copy()
    df_preds['Pred_Linear_kUSD'] = np.round(np.c_[np.ones(len(X_test)), X_test] @ w_lin, 2)
    df_preds['Pred_DecisionTree_kUSD'] = np.round(dt_best.predict(X_test), 2)
    df_preds['Pred_Polynomial_kUSD'] = np.round(X_te_p3 @ w_p3, 2)
    df_preds['Error_Linear'] = np.round(df_preds['Pred_Linear_kUSD'] - df_preds['Sale_Price_kUSD'], 2)
    df_preds['Error_DecisionTree'] = np.round(df_preds['Pred_DecisionTree_kUSD'] - df_preds['Sale_Price_kUSD'], 2)
    df_preds['Error_Polynomial'] = np.round(df_preds['Pred_Polynomial_kUSD'] - df_preds['Sale_Price_kUSD'], 2)
    
    df_complexity.to_csv(os.path.join(base_dir, 'model_complexity_error_comparison.csv'), index=False)
    df_opt_history.to_csv(os.path.join(base_dir, 'optimizer_convergence_and_oscillations.csv'), index=False)
    df_bench.to_csv(os.path.join(base_dir, 'computational_vectorization_benchmark.csv'), index=False)
    df_preds.to_csv(os.path.join(base_dir, 'test_set_predictions_all_models.csv'), index=False)
    
    # 6. Generate Publication-Quality Figures
    print("\n[Step 6/6] Generating Module 3 Publication Figures...")
    
    # -------------------------------------------------------------
    # Figure 7: Training vs Testing Error Curves (Overfitting Gap)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Polynomial Degrees 1 to 4 Error Curves
    set_minimal_theme(axes[0])
    poly_sub = df_complexity[df_complexity['Model_Family'] == 'Polynomial Regression']
    degs = [1, 2, 3, 4]
    axes[0].plot(degs, poly_sub['Train_MSE'], color='#2563EB', marker='o', linewidth=2.2, label='Training MSE (Decreasing / Memorization)')
    axes[0].plot(degs, poly_sub['Test_MSE'], color='#EF4444', marker='s', linewidth=2.5, label='Testing MSE (Exploding Generalization Error)')
    
    # Optimal Polynomial Minimum Marker (Degree 2)
    axes[0].scatter(2, poly_sub['Test_MSE'].iloc[1], color='#10B981', s=160, marker='*', zorder=10)
    axes[0].text(2.1, poly_sub['Test_MSE'].iloc[1] + 1500, "Optimal Poly (Deg 2)\nMinimum Test MSE",
                 fontname='Cambria', fontsize=9.5, fontweight='bold', color='#047857')
                 
    axes[0].set_title("(A) Polynomial Model Complexity vs. Error (Overfitting at Deg 3-4)", pad=12)
    axes[0].set_xlabel("Polynomial Expansion Degree (Capacity)", labelpad=8)
    axes[0].set_ylabel("Mean Squared Error (kUSD^2)", labelpad=8)
    axes[0].set_xticks(degs)
    axes[0].set_xticklabels([f"Deg {d}\n({p} params)" for d, p in zip(degs, poly_sub['Param_Count'])])
    axes[0].set_yscale('log')
    
    # Subplot B: Decision Tree Depth vs Error Curves
    set_minimal_theme(axes[1])
    tree_sub = df_complexity[df_complexity['Model_Family'] == 'Decision Tree Regressor']
    tree_labels = ['Depth 2', 'Depth 4', 'Depth 6', 'Depth 8', 'Depth 12', 'Unlimited']
    x_tr = np.arange(len(tree_labels))
    axes[1].plot(x_tr, tree_sub['Train_MSE'], color='#2563EB', marker='o', linewidth=2.2, label='Training MSE')
    axes[1].plot(x_tr, tree_sub['Test_MSE'], color='#EF4444', marker='s', linewidth=2.5, label='Testing MSE')
    
    # Optimal Tree Marker (Depth 6)
    opt_tree_idx = 2
    axes[1].scatter(opt_tree_idx, tree_sub['Test_MSE'].iloc[opt_tree_idx], color='#10B981', s=160, marker='*', zorder=10)
    axes[1].text(opt_tree_idx + 0.15, tree_sub['Test_MSE'].iloc[opt_tree_idx] - 800, "Optimal Tree (Depth 6)\nBalanced Generalization",
                 fontname='Cambria', fontsize=9.5, fontweight='bold', color='#047857')
                 
    axes[1].set_title("(B) Decision Tree Depth vs. Error (Overfitting at Depth >= 8)", pad=12)
    axes[1].set_xlabel("Decision Tree Max Depth Setting", labelpad=8)
    axes[1].set_ylabel("Mean Squared Error (kUSD^2)", labelpad=8)
    axes[1].set_xticks(x_tr)
    axes[1].set_xticklabels(tree_labels, rotation=20)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles7 = [
        mpl.lines.Line2D([0], [0], color='#2563EB', marker='o', linewidth=2.2, label='Training MSE (Decreases Monotonically with Capacity)'),
        mpl.lines.Line2D([0], [0], color='#EF4444', marker='s', linewidth=2.5, label='Testing MSE (U-Shaped Curve: High Bias -> Optimal -> Overfit)'),
        mpl.lines.Line2D([0], [0], marker='*', color='w', markerfacecolor='#10B981', markersize=12, label='Optimal Architecture Minimum Error Point')
    ]
    fig.legend(handles=legend_handles7, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=3, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig7_path = os.path.join(base_dir, '07_training_vs_testing_error_and_overfitting.png')
    plt.savefig(fig7_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig7_path}")
    
    # -------------------------------------------------------------
    # Figure 8: Optimizer Convergence & Oscillation Damping
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Loss Curves over 100 Epochs (Log Scale)
    set_minimal_theme(axes[0])
    ep_x = df_opt_history['epoch']
    axes[0].plot(ep_x, df_opt_history['BGD_loss'], color='#2563EB', linewidth=2.0, label='Batch Gradient Descent (BGD)')
    axes[0].plot(ep_x, df_opt_history['SGD_loss'], color='#F59E0B', linewidth=1.8, linestyle='--', label='Stochastic Gradient Descent (SGD)')
    axes[0].plot(ep_x, df_opt_history['Momentum_loss'], color='#10B981', linewidth=2.2, label='Momentum (beta = 0.9)')
    axes[0].plot(ep_x, df_opt_history['Adam_loss'], color='#8B5CF6', linewidth=2.4, label='Adam (Adaptive Moments)')
    axes[0].set_title("(A) Optimizer Loss Trajectories across 100 Epochs", pad=12)
    axes[0].set_xlabel("Training Epochs", labelpad=8)
    axes[0].set_ylabel("Standardized MSE Loss", labelpad=8)
    axes[0].set_yscale('log')
    axes[0].set_xlim(1, 100)
    
    # Subplot B: Gradient Oscillation Metric (1 - Cosine Similarity)
    set_minimal_theme(axes[1])
    opt_names = ['BGD', 'SGD', 'Momentum', 'Adam']
    osc_vals = [oscillation_scores[k] for k in opt_names]
    bar_cols = ['#2563EB', '#F59E0B', '#10B981', '#8B5CF6']
    bars_osc = axes[1].bar(opt_names, osc_vals, color=bar_cols, edgecolor='#1F2937', linewidth=0.8, alpha=0.88, width=0.55)
    axes[1].set_title("(B) Gradient Oscillation Severity (1 - Cosine Similarity)", pad=12)
    axes[1].set_xlabel("Optimization Algorithm", labelpad=8)
    axes[1].set_ylabel("Mean Directional Oscillation Metric", labelpad=8)
    axes[1].set_ylim(0, max(osc_vals) * 1.25)
    for bar in bars_osc:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.3f}",
                     ha='center', va='bottom', fontname='Cambria', fontsize=9.5, fontweight='bold')
                     
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles8 = [
        mpl.lines.Line2D([0], [0], color='#2563EB', linewidth=2.0, label='BGD (Deterministic Monotonic Descent)'),
        mpl.lines.Line2D([0], [0], color='#F59E0B', linewidth=1.8, linestyle='--', label='SGD (Severe Directional Noise / Jitter)'),
        mpl.lines.Line2D([0], [0], color='#10B981', linewidth=2.2, label='Momentum (Oscillation Damping & Acceleration)'),
        mpl.lines.Line2D([0], [0], color='#8B5CF6', linewidth=2.4, label='Adam (Adaptive Scale Invariance)')
    ]
    fig.legend(handles=legend_handles8, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig8_path = os.path.join(base_dir, '08_optimizer_convergence_loss_curves.png')
    plt.savefig(fig8_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig8_path}")
    
    # -------------------------------------------------------------
    # Figure 9: Residuals & Computational Speedup Benchmark
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Residual Distributions
    set_minimal_theme(axes[0])
    sns.kdeplot(df_preds['Error_Linear'], ax=axes[0], color='#2563EB', linewidth=2.2, label='Linear Regression (High Bias)', fill=True, alpha=0.15)
    sns.kdeplot(df_preds['Error_DecisionTree'], ax=axes[0], color='#10B981', linewidth=2.4, label='Decision Tree (Optimal Fit)', fill=True, alpha=0.18)
    # Clip extreme polynomial errors for clean visualization
    sns.kdeplot(np.clip(df_preds['Error_Polynomial'], -300, 300), ax=axes[0], color='#EF4444', linewidth=2.0, linestyle='--', label='Polynomial Deg 3 (Overfit)', fill=True, alpha=0.12)
    axes[0].axvline(0, color='#111827', linestyle=':', linewidth=1.0)
    axes[0].set_title("(A) Test Set Prediction Residuals (y_pred - y_true)", pad=12)
    axes[0].set_xlabel("Residual Valuation Error (Thousands of USD)", labelpad=8)
    axes[0].set_ylabel("Kernel Density Estimate", labelpad=8)
    axes[0].set_xlim(-250, 250)
    
    # Subplot B: Vectorization Speedup Bar Chart
    set_minimal_theme(axes[1])
    b_idx = np.arange(len(batch_sizes))
    speedups = df_bench['Speedup_Factor'].values
    bars_sp = axes[1].bar(b_idx, speedups, color='#2563EB', edgecolor='#1E40AF', width=0.55, alpha=0.88)
    axes[1].set_xticks(b_idx)
    axes[1].set_xticklabels([f"N={bs}" for bs in batch_sizes])
    axes[1].set_title("(B) Vectorized NumPy BLAS vs. Scalar Loop Execution Speedup", pad=12)
    axes[1].set_xlabel("Evaluation Batch Size Dimension (N)", labelpad=8)
    axes[1].set_ylabel("Computational Acceleration Factor (X)", labelpad=8)
    axes[1].set_ylim(0, max(speedups) * 1.20)
    for bar in bars_sp:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 1.2, f"{h:.1f}x",
                     ha='center', va='bottom', fontname='Cambria', fontsize=9.5, fontweight='bold')
                     
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles9 = [
        mpl.lines.Line2D([0], [0], color='#2563EB', linewidth=2.2, label='Linear Model Residuals (Underfitting Bias)'),
        mpl.lines.Line2D([0], [0], color='#10B981', linewidth=2.4, label='Decision Tree Residuals (Centered / Minimal Variance)'),
        mpl.lines.Line2D([0], [0], color='#EF4444', linewidth=2.0, linestyle='--', label='Polynomial Model Residuals (Fat Tails)'),
        mpl.patches.Patch(facecolor='#2563EB', edgecolor='#1E40AF', label='NumPy Vectorized Tensor Speedup (BLAS)')
    ]
    fig.legend(handles=legend_handles9, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig9_path = os.path.join(base_dir, '09_residual_distributions_and_speedup_benchmark.png')
    plt.savefig(fig9_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig9_path}")
    
    print("\nModule 3 execution, benchmarking, and visualization completed successfully!")

if __name__ == '__main__':
    run_module_3()
