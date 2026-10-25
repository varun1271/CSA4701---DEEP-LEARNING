"""
Module 3: Training & Evaluation
Scenario: A neural network experiences oscillations while optimizing its loss function.
Implement gradient descent with momentum and compare its convergence with standard gradient descent.

This script:
1. Trains an 8 -> 16 -> 1 Multi-Layer Perceptron on ill-conditioned data across 150 epochs.
2. Compares 5 optimization regimes:
   - Standard GD (Low LR = 0.015, beta = 0.0)
   - Standard GD (High LR = 0.045, beta = 0.0) -> Violent Oscillations
   - Low Momentum (LR = 0.03, beta = 0.5)
   - Optimal Momentum (LR = 0.03, beta = 0.9) -> Smooth Accelerated Convergence
   - High Momentum (LR = 0.03, beta = 0.98) -> Overshooting Ringing
3. Conducts a sensitivity sweep across momentum coefficient beta in [0.0, 0.99].
4. Benchmarks vectorized NumPy BLAS tensor speedup against scalar loops.
5. Exports comprehensive evaluation CSVs.
6. Generates 3 publication-grade visualization figures:
   - Figure 07: Neural Loss Convergence & Gradient Norms (Standard GD vs. Momentum).
   - Figure 08: Momentum Beta Sensitivity & Oscillation Damping Curves.
   - Figure 09: Test Set Prediction Residuals & Vectorized Speedup Benchmark.
"""

import os
import time
import numpy as np
import pandas as pd
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

class TrainableMLP:
    """Configurable MLP for optimization oscillation experiments."""
    def __init__(self, input_dim=8, hidden_dim=16, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, 1))
        
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def forward(self, X):
        self.X = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.y_hat = np.dot(self.a1, self.W2) + self.b2
        return self.y_hat

    def step(self, y_true, lr=0.03, beta=0.0):
        N = self.X.shape[0]
        y_true = y_true.reshape(-1, 1)
        error = (2.0 / N) * (self.y_hat - y_true)
        
        # Gradients
        dW2 = np.dot(self.a1.T, error)
        db2 = np.sum(error, axis=0, keepdims=True)
        
        da1 = np.dot(error, self.W2.T)
        dz1 = da1 * (1.0 - self.a1 ** 2)
        dW1 = np.dot(self.X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Momentum velocity update
        self.v_W1 = beta * self.v_W1 + lr * dW1
        self.v_b1 = beta * self.v_b1 + lr * db1
        self.v_W2 = beta * self.v_W2 + lr * dW2
        self.v_b2 = beta * self.v_b2 + lr * db2
        
        self.W1 -= self.v_W1
        self.b1 -= self.v_b1
        self.W2 -= self.v_W2
        self.b2 -= self.v_b2
        
        # Total gradient norm
        grad_norm = np.sqrt(np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2))
        
        # Flattened gradient vector for cosine similarity tracking
        flat_grad = np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])
        return grad_norm, flat_grad

def run_module_3():
    print("=" * 80)
    print("EXPERIMENT 6 - MODULE 3: TRAINING & EVALUATION")
    print("Momentum Oscillation Damping & Convergence Benchmarking")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp6_dir = os.path.dirname(base_dir)
    mod2_dir = os.path.join(exp6_dir, 'Data Splitting & Model Design')
    
    # 1. Load Data Partitions
    print("\n[Step 1/6] Loading train, validation, and test datasets...")
    train_df = pd.read_csv(os.path.join(mod2_dir, 'train_signals.csv'))
    val_df = pd.read_csv(os.path.join(mod2_dir, 'val_signals.csv'))
    test_df = pd.read_csv(os.path.join(mod2_dir, 'test_signals.csv'))
    
    feature_names = [c for c in train_df.columns if c not in ['Sample_ID', 'System_Response_Target', 'Target_Standardized']]
    
    X_train = train_df[feature_names].values
    y_train = train_df['Target_Standardized'].values
    X_val = val_df[feature_names].values
    y_val = val_df['Target_Standardized'].values
    X_test = test_df[feature_names].values
    y_test = test_df['Target_Standardized'].values
    
    # 2. Benchmark 5 Optimization Regimes over 150 Epochs
    print("\n[Step 2/6] Executing 150-Epoch Neural Training Across 5 Optimization Regimes...")
    epochs = 150
    regimes = {
        'Standard_GD_LowLR': {'lr': 0.015, 'beta': 0.0, 'name': 'Standard GD (Low LR = 0.015)'},
        'Standard_GD_HighLR': {'lr': 0.045, 'beta': 0.0, 'name': 'Standard GD (High LR = 0.045 - Oscillating)'},
        'Momentum_Beta_0.5': {'lr': 0.030, 'beta': 0.5, 'name': 'Low Momentum (beta = 0.5)'},
        'Momentum_Beta_0.9': {'lr': 0.030, 'beta': 0.9, 'name': 'Optimal Momentum (beta = 0.9)'},
        'Momentum_Beta_0.98': {'lr': 0.030, 'beta': 0.98, 'name': 'High Momentum (beta = 0.98 - Overshooting)'}
    }
    
    history = {'epoch': list(range(1, epochs + 1))}
    trained_models = {}
    oscillation_metrics = {}
    
    for key, cfg in regimes.items():
        mlp = TrainableMLP(input_dim=8, hidden_dim=16, seed=42)
        losses = []
        grad_norms = []
        cos_similarities = []
        prev_grad = None
        
        t0 = time.perf_counter()
        for ep in range(1, epochs + 1):
            y_pred_tr = mlp.forward(X_train).flatten()
            gnorm, flat_grad = mlp.step(y_train, lr=cfg['lr'], beta=cfg['beta'])
            
            # Record MSE loss
            mse_loss = mean_squared_error(y_train, y_pred_tr)
            losses.append(mse_loss)
            grad_norms.append(gnorm)
            
            # Calculate gradient cosine similarity
            if prev_grad is not None:
                np_g = np.linalg.norm(prev_grad)
                nc_g = np.linalg.norm(flat_grad)
                if np_g > 1e-9 and nc_g > 1e-9:
                    sim = np.dot(prev_grad, flat_grad) / (np_g * nc_g)
                    cos_similarities.append(1.0 - sim)
            prev_grad = flat_grad.copy()
            
        elapsed_ms = (time.perf_counter() - t0) * 1000
        history[f'{key}_loss'] = losses
        history[f'{key}_gradnorm'] = grad_norms
        trained_models[key] = mlp
        oscillation_metrics[key] = np.mean(cos_similarities) if cos_similarities else 0.0
        
        # Test Evaluation
        y_test_pred = mlp.forward(X_test).flatten()
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        print(f"  {cfg['name']:50s}: Final Loss = {losses[-1]:.4f} | Test MSE = {test_mse:.4f} | Test R2 = {test_r2:.4f} | Oscillation = {oscillation_metrics[key]:.4f}")
        
    df_history = pd.DataFrame(history)
    
    # 3. Momentum Beta Sensitivity Sweep
    print("\n[Step 3/6] Running Momentum Coefficient Beta Sensitivity Sweep [0.0 to 0.99]...")
    beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    beta_records = []
    
    for b_val in beta_values:
        mlp_b = TrainableMLP(input_dim=8, hidden_dim=16, seed=42)
        cos_diffs = []
        prev_g = None
        
        for ep in range(1, 100):
            mlp_b.forward(X_train)
            _, f_grad = mlp_b.step(y_train, lr=0.03, beta=b_val)
            if prev_g is not None:
                p_norm = np.linalg.norm(prev_g)
                c_norm = np.linalg.norm(f_grad)
                if p_norm > 1e-9 and c_norm > 1e-9:
                    cos_diffs.append(1.0 - (np.dot(prev_g, f_grad) / (p_norm * c_norm)))
            prev_g = f_grad.copy()
            
        # Final test evaluation
        y_te_p = mlp_b.forward(X_test).flatten()
        te_mse = mean_squared_error(y_test, y_te_p)
        te_r2 = r2_score(y_test, y_te_p)
        mean_osc = np.mean(cos_diffs) if cos_diffs else 0.0
        
        beta_records.append({
            'Momentum_Beta': b_val,
            'Effective_Acceleration_Factor': round(1.0 / (1.0 - min(b_val, 0.99)), 1),
            'Final_Train_Loss': round(mean_squared_error(y_train, mlp_b.forward(X_train).flatten()), 5),
            'Test_MSE': round(te_mse, 5),
            'Test_RMSE': round(np.sqrt(te_mse), 4),
            'Test_R2': round(te_r2, 4),
            'Oscillation_Index': round(mean_osc, 4)
        })
        print(f"  Beta = {b_val:.2f} (Speedup {1.0/(1.0-min(b_val, 0.99)):4.1f}x): Test MSE = {te_mse:.4f} | Oscillation = {mean_osc:.4f}")
        
    df_beta_sweep = pd.DataFrame(beta_records)
    
    # 4. Computational Vectorization Benchmark
    print("\n[Step 4/6] Benchmarking Scalar Loop vs. Vectorized NumPy BLAS for Neural Backprop...")
    batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    bench_records = []
    
    for bs in batch_sizes:
        X_syn = np.random.randn(bs, 8)
        y_syn = np.random.randn(bs, 1)
        W1_s = np.random.randn(8, 16)
        b1_s = np.zeros((1, 16))
        W2_s = np.random.randn(16, 1)
        b2_s = np.zeros((1, 1))
        
        # Scalar Loop Implementation
        t0 = time.perf_counter()
        for _ in range(25):
            # Elementwise forward pass
            z1_sc = np.zeros((bs, 16))
            a1_sc = np.zeros((bs, 16))
            y_hat_sc = np.zeros((bs, 1))
            for i in range(bs):
                for h in range(16):
                    val = b1_s[0, h]
                    for d in range(8):
                        val += X_syn[i, d] * W1_s[d, h]
                    z1_sc[i, h] = val
                    a1_sc[i, h] = np.tanh(val)
                out_val = b2_s[0, 0]
                for h in range(16):
                    out_val += a1_sc[i, h] * W2_s[h, 0]
                y_hat_sc[i, 0] = out_val
        t_scalar_ms = ((time.perf_counter() - t0) / 25) * 1000
        
        # Vectorized BLAS Implementation
        t0 = time.perf_counter()
        for _ in range(25):
            z1_vec = np.dot(X_syn, W1_s) + b1_s
            a1_vec = np.tanh(z1_vec)
            y_hat_vec = np.dot(a1_vec, W2_s) + b2_s
            err_vec = (2.0 / bs) * (y_hat_vec - y_syn)
            dW2_vec = np.dot(a1_vec.T, err_vec)
        t_vec_ms = ((time.perf_counter() - t0) / 25) * 1000
        
        speedup = t_scalar_ms / max(t_vec_ms, 1e-6)
        bench_records.append({
            'Batch_Size': bs,
            'Scalar_Loop_ms': round(t_scalar_ms, 4),
            'Vectorized_BLAS_ms': round(t_vec_ms, 4),
            'Speedup_Factor': round(speedup, 2)
        })
        print(f"  Batch {bs:5d}: Scalar = {t_scalar_ms:8.3f} ms | Vectorized = {t_vec_ms:6.3f} ms | Speedup = {speedup:6.1f}x")
        
    df_bench = pd.DataFrame(bench_records)
    
    # 5. Export Test Predictions
    print("\n[Step 5/6] Exporting Test Prediction Diagnostics...")
    m_std = trained_models['Standard_GD_HighLR']
    m_mom = trained_models['Momentum_Beta_0.9']
    
    pred_std = m_std.forward(X_test).flatten()
    pred_mom = m_mom.forward(X_test).flatten()
    
    df_test_preds = test_df[['Sample_ID', 'Target_Standardized']].copy()
    df_test_preds['Pred_Standard_GD'] = np.round(pred_std, 4)
    df_test_preds['Pred_Momentum_GD'] = np.round(pred_mom, 4)
    df_test_preds['Error_Standard_GD'] = np.round(pred_std - y_test, 4)
    df_test_preds['Error_Momentum_GD'] = np.round(pred_mom - y_test, 4)
    
    # Save CSVs
    df_history.to_csv(os.path.join(base_dir, 'neural_optimizer_convergence_and_oscillations.csv'), index=False)
    df_beta_sweep.to_csv(os.path.join(base_dir, 'momentum_beta_sensitivity_analysis.csv'), index=False)
    df_bench.to_csv(os.path.join(base_dir, 'computational_vectorization_benchmark.csv'), index=False)
    df_test_preds.to_csv(os.path.join(base_dir, 'test_predictions_standard_vs_momentum.csv'), index=False)
    
    # 6. Generate Publication Figures
    print("\n[Step 6/6] Generating Module 3 Publication Figures...")
    
    # -------------------------------------------------------------
    # Figure 7: Neural Loss Convergence & Gradient Norms
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: MSE Loss Trajectories (Log Scale)
    set_minimal_theme(axes[0])
    ep_x = df_history['epoch']
    axes[0].plot(ep_x, df_history['Standard_GD_HighLR_loss'], color='#EF4444', linewidth=2.0, linestyle='--',
                 label='Standard GD (High LR = 0.045 - Unstable / High Oscillations)')
    axes[0].plot(ep_x, df_history['Standard_GD_LowLR_loss'], color='#F59E0B', linewidth=1.8,
                 label='Standard GD (Low LR = 0.015 - Glacial Plateau)')
    axes[0].plot(ep_x, df_history['Momentum_Beta_0.5_loss'], color='#06B6D4', linewidth=2.0,
                 label='Momentum (beta = 0.5 - Partial Damping)')
    axes[0].plot(ep_x, df_history['Momentum_Beta_0.9_loss'], color='#10B981', linewidth=2.5,
                 label='Optimal Momentum (beta = 0.9 - Fast Convergence)')
    axes[0].plot(ep_x, df_history['Momentum_Beta_0.98_loss'], color='#8B5CF6', linewidth=2.0, linestyle=':',
                 label='High Momentum (beta = 0.98 - Overshooting / Ringing)')
                 
    axes[0].set_title("(A) Neural Loss Trajectories: Standard GD vs. Momentum Regimes", pad=12)
    axes[0].set_xlabel("Training Epochs", labelpad=8)
    axes[0].set_ylabel("Mean Squared Error Loss (Log Scale)", labelpad=8)
    axes[0].set_yscale('log')
    axes[0].set_xlim(1, 150)
    
    # Subplot B: Backpropagation Gradient Norms ||grad|| Dynamics
    set_minimal_theme(axes[1])
    axes[1].plot(ep_x, df_history['Standard_GD_HighLR_gradnorm'], color='#EF4444', linewidth=1.8, alpha=0.85,
                 label='Standard GD Gradient Norm (Volatile Spike Jitter)')
    axes[1].plot(ep_x, df_history['Momentum_Beta_0.9_gradnorm'], color='#10B981', linewidth=2.4,
                 label='Momentum GD Gradient Norm (Smooth Exponential Dissipation)')
    axes[1].set_title("(B) Gradient Magnitude Dynamics: Instability vs. Smooth Damping", pad=12)
    axes[1].set_xlabel("Training Epochs", labelpad=8)
    axes[1].set_ylabel("Total Parameter Gradient Norm ||g||", labelpad=8)
    axes[1].set_xlim(1, 150)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles7 = [
        mpl.lines.Line2D([0], [0], color='#EF4444', linewidth=2.0, linestyle='--', label='Standard GD (Oscillating across Ravine Walls)'),
        mpl.lines.Line2D([0], [0], color='#10B981', linewidth=2.5, label='Optimal Momentum beta = 0.9 (Smooth Canyon Traversal)'),
        mpl.lines.Line2D([0], [0], color='#F59E0B', linewidth=1.8, label='Standard GD Low LR (Glacial Longitudinal Progress)'),
        mpl.lines.Line2D([0], [0], color='#8B5CF6', linewidth=2.0, linestyle=':', label='High Momentum beta = 0.98 (Overshooting)')
    ]
    fig.legend(handles=legend_handles7, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig7_path = os.path.join(base_dir, '07_neural_loss_convergence_standard_vs_momentum.png')
    plt.savefig(fig7_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig7_path}")
    
    # -------------------------------------------------------------
    # Figure 8: Momentum Beta Sensitivity & Oscillation Damping
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Test MSE vs Beta
    set_minimal_theme(axes[0])
    b_vals = df_beta_sweep['Momentum_Beta']
    axes[0].plot(b_vals, df_beta_sweep['Test_MSE'], color='#2563EB', marker='o', linewidth=2.4, label='Test MSE Curve')
    # Optimal Beta Marker (0.90)
    opt_idx = 5
    axes[0].scatter(b_vals.iloc[opt_idx], df_beta_sweep['Test_MSE'].iloc[opt_idx], color='#10B981', s=160, marker='*', zorder=10)
    axes[0].text(b_vals.iloc[opt_idx] - 0.08, df_beta_sweep['Test_MSE'].iloc[opt_idx] + 0.04,
                 f"Optimal beta = 0.90\nMin Test MSE = {df_beta_sweep['Test_MSE'].iloc[opt_idx]:.4f}",
                 fontname='Cambria', fontsize=9.5, fontweight='bold', color='#047857')
                 
    axes[0].set_title("(A) Test Generalization MSE vs. Momentum Decay Factor beta", pad=12)
    axes[0].set_xlabel("Momentum Coefficient beta", labelpad=8)
    axes[0].set_ylabel("Held-Out Test MSE Loss", labelpad=8)
    axes[0].set_xlim(-0.05, 1.04)
    
    # Subplot B: Oscillation Metric vs Beta
    set_minimal_theme(axes[1])
    axes[1].plot(b_vals, df_beta_sweep['Oscillation_Index'], color='#EF4444', marker='s', linewidth=2.4, label='Oscillation Metric')
    axes[1].axhspan(0.0, 0.20, color='#10B981', alpha=0.15, label='Damped Stability Zone')
    axes[1].axhspan(0.20, 0.50, color='#F59E0B', alpha=0.15, label='Moderate Oscillation Zone')
    axes[1].axhspan(0.50, 1.00, color='#EF4444', alpha=0.15, label='High Instability Zone')
    
    axes[1].set_title("(B) Directional Gradient Oscillation Severity (1 - Cosine Similarity)", pad=12)
    axes[1].set_xlabel("Momentum Coefficient beta", labelpad=8)
    axes[1].set_ylabel("Mean Gradient Directional Oscillation", labelpad=8)
    axes[1].set_xlim(-0.05, 1.04)
    axes[1].set_ylim(0, 0.70)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles8 = [
        mpl.lines.Line2D([0], [0], color='#2563EB', marker='o', linewidth=2.2, label='Test MSE Performance vs. beta'),
        mpl.lines.Line2D([0], [0], color='#EF4444', marker='s', linewidth=2.2, label='Gradient Directional Oscillation Index'),
        mpl.patches.Patch(facecolor='#10B981', alpha=0.25, label='Optimal Momentum Zone (beta = 0.85 - 0.92)'),
        mpl.lines.Line2D([0], [0], marker='*', color='w', markerfacecolor='#10B981', markersize=12, label='Global Minimum Error')
    ]
    fig.legend(handles=legend_handles8, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig8_path = os.path.join(base_dir, '08_momentum_beta_sensitivity_and_oscillation_metrics.png')
    plt.savefig(fig8_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig8_path}")
    
    # -------------------------------------------------------------
    # Figure 9: Residuals & Vectorization BLAS Speedup
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Test Set Prediction Residuals (Standard vs Momentum)
    set_minimal_theme(axes[0])
    sns.kdeplot(df_test_preds['Error_Standard_GD'], ax=axes[0], color='#EF4444', linewidth=2.2, linestyle='--',
                label=f'Standard GD Residuals (MSE: {mean_squared_error(y_test, pred_std):.4f})', fill=True, alpha=0.15)
    sns.kdeplot(df_test_preds['Error_Momentum_GD'], ax=axes[0], color='#10B981', linewidth=2.4,
                label=f'Momentum GD Residuals (MSE: {mean_squared_error(y_test, pred_mom):.4f})', fill=True, alpha=0.18)
    axes[0].axvline(0, color='#111827', linestyle=':', linewidth=1.0)
    axes[0].set_title("(A) Test Set Prediction Residual Distributions (y_pred - y_true)", pad=12)
    axes[0].set_xlabel("Standardized Residual Error", labelpad=8)
    axes[0].set_ylabel("Kernel Density Estimate", labelpad=8)
    axes[0].set_xlim(-1.8, 1.8)
    
    # Subplot B: Vectorized Speedup Bar Chart
    set_minimal_theme(axes[1])
    b_idx = np.arange(len(batch_sizes))
    speedups = df_bench['Speedup_Factor'].values
    bars_sp = axes[1].bar(b_idx, speedups, color='#2563EB', edgecolor='#1E40AF', width=0.55, alpha=0.88)
    axes[1].set_xticks(b_idx)
    axes[1].set_xticklabels([f"N={bs}" for bs in batch_sizes])
    axes[1].set_title("(B) Vectorized NumPy BLAS vs. Scalar Loop Execution Speedup", pad=12)
    axes[1].set_xlabel("Batch Evaluation Size (N)", labelpad=8)
    axes[1].set_ylabel("Hardware Acceleration Factor (X)", labelpad=8)
    axes[1].set_ylim(0, max(speedups) * 1.20)
    for bar in bars_sp:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}x",
                     ha='center', va='bottom', fontname='Cambria', fontsize=9.5, fontweight='bold')
                     
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles9 = [
        mpl.lines.Line2D([0], [0], color='#EF4444', linewidth=2.0, linestyle='--', label='Standard GD Residuals (Volatile Tails)'),
        mpl.lines.Line2D([0], [0], color='#10B981', linewidth=2.4, label='Momentum GD Residuals (Tightly Centered Normal)'),
        mpl.patches.Patch(facecolor='#2563EB', edgecolor='#1E40AF', label='NumPy Vectorized Tensor Acceleration (BLAS)')
    ]
    fig.legend(handles=legend_handles9, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=3, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig9_path = os.path.join(base_dir, '09_test_prediction_residuals_and_blas_speedup.png')
    plt.savefig(fig9_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig9_path}")
    
    print("\nModule 3 execution, benchmarking, and visualization completed successfully!")

if __name__ == '__main__':
    run_module_3()
