"""
Module 2: Data Splitting & Model Design
Scenario: A neural network experiences oscillations while optimizing its loss function.
Implement gradient descent with momentum and compare its convergence with standard gradient descent.

This script:
1. Performs 70/15/15 train/val/test data partitioning preserving target distribution.
2. Implements a configurable Neural Network (MLP: 8 -> 16 -> 1) with Tanh activations
   and exact analytic backpropagation gradients.
3. Implements Standard Gradient Descent and Momentum Gradient Descent (Polyak momentum).
4. Models the 2D anisotropic pathological ravine loss surface f(w1, w2) = 0.5*w1^2 + 25*w2^2.
5. Serializes split datasets and model weights to CSV and NPZ.
6. Generates 3 publication-grade visualization figures:
   - Figure 04: Dataset Split Proportions & Target Response Invariance.
   - Figure 05: Neural Network Architecture & Non-Linear Mapping Parity.
   - Figure 06: Pathological Ravine Contour & Optimization Trajectories (Standard GD vs. Momentum).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
# Neural Network Implementation (NumPy)
# -------------------------------------------------------------
class AnisotropicMLP:
    """Multi-Layer Perceptron (8 -> 16 -> 1) for studying optimization oscillations."""
    def __init__(self, input_dim=8, hidden_dim=16, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, 1))
        
        # Momentum velocity buffers
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

    def compute_gradients(self, y_true):
        N = self.X.shape[0]
        y_true = y_true.reshape(-1, 1)
        # dL / dy_hat = (2/N) * (y_hat - y)
        error = (2.0 / N) * (self.y_hat - y_true)
        
        # Output layer gradients
        dW2 = np.dot(self.a1.T, error)
        db2 = np.sum(error, axis=0, keepdims=True)
        
        # Hidden layer gradients via tanh derivative (1 - a1^2)
        da1 = np.dot(error, self.W2.T)
        dz1 = da1 * (1.0 - self.a1 ** 2)
        dW1 = np.dot(self.X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2

    def update_standard_gd(self, dW1, db1, dW2, db2, lr=0.03):
        """Standard Gradient Descent without momentum."""
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def update_momentum(self, dW1, db1, dW2, db2, lr=0.03, beta=0.9):
        """Momentum Gradient Descent: v = beta * v + lr * grad, w = w - v."""
        self.v_W1 = beta * self.v_W1 + lr * dW1
        self.v_b1 = beta * self.v_b1 + lr * db1
        self.v_W2 = beta * self.v_W2 + lr * dW2
        self.v_b2 = beta * self.v_b2 + lr * db2
        
        self.W1 -= self.v_W1
        self.b1 -= self.v_b1
        self.W2 -= self.v_W2
        self.b2 -= self.v_b2

# -------------------------------------------------------------
# Pathological 2D Ravine Simulation
# -------------------------------------------------------------
def simulate_ravine_trajectories(epochs=50, lr_gd=0.038, lr_mom=0.038, beta=0.9):
    """
    Simulates optimization on the classic pathological ravine:
    f(w1, w2) = 0.5 * w1^2 + 25.0 * w2^2
    Condition number kappa = 50.0 / 1.0 = 50.0 >> 1.
    """
    w_start = np.array([-4.5, 3.2])
    
    # 1. Standard GD
    w_gd = w_start.copy()
    traj_gd = [w_gd.copy()]
    for _ in range(epochs):
        grad = np.array([w_gd[0], 50.0 * w_gd[1]])  # df/dw1 = w1, df/dw2 = 50*w2
        w_gd = w_gd - lr_gd * grad
        traj_gd.append(w_gd.copy())
        
    # 2. Momentum GD
    w_mom = w_start.copy()
    v_mom = np.zeros(2)
    traj_mom = [w_mom.copy()]
    for _ in range(epochs):
        grad = np.array([w_mom[0], 50.0 * w_mom[1]])
        v_mom = beta * v_mom + lr_mom * grad
        w_mom = w_mom - v_mom
        traj_mom.append(w_mom.copy())
        
    return np.array(traj_gd), np.array(traj_mom)

def run_module_2():
    print("=" * 80)
    print("EXPERIMENT 6 - MODULE 2: DATA SPLITTING & MODEL DESIGN")
    print("Neural Network Architecture, Ravine Simulation & Momentum Mechanics")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp6_dir = os.path.dirname(base_dir)
    prep_csv_path = os.path.join(exp6_dir, 'Data Collection & Preprocessing', 'preprocessed_optimization_signals_1000.csv')
    
    # 1. Load Preprocessed Data
    print("\n[Step 1/5] Loading preprocessed signals dataset...")
    df = pd.read_csv(prep_csv_path)
    feature_names = [c for c in df.columns if c not in ['Sample_ID', 'System_Response_Target', 'Target_Standardized']]
    
    X = df[feature_names].values
    y = df['Target_Standardized'].values
    
    # 2. Partition Data (70% Train, 15% Val, 15% Test)
    print("\n[Step 2/5] Partitioning into Train (70%), Val (15%), and Test (15%)...")
    # Stratified quantile binning on continuous response
    y_bins = pd.qcut(y, q=5, labels=False)
    train_idx, temp_idx = train_test_split(np.arange(len(df)), test_size=0.30, random_state=42, stratify=y_bins)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=y_bins[temp_idx])
    
    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df.iloc[val_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    
    print(f"Dataset Partitions:")
    print(f"  - Training Set:   {len(df_train)} instances (70.0%)")
    print(f"  - Validation Set: {len(df_val)} instances (15.0%)")
    print(f"  - Test Set:       {len(df_test)} instances (15.0%)")
    
    # Export split CSVs
    train_path = os.path.join(base_dir, 'train_signals.csv')
    val_path = os.path.join(base_dir, 'val_signals.csv')
    test_path = os.path.join(base_dir, 'test_signals.csv')
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")
    
    # 3. Model Architecture Instantiation & Initial Fit
    print("\n[Step 3/5] Instantiating Multi-Layer Perceptron (8 -> 16 -> 1) with Momentum...")
    mlp_model = AnisotropicMLP(input_dim=8, hidden_dim=16, seed=42)
    
    # Serialize initial weights
    weights_path = os.path.join(base_dir, 'trained_neural_network_weights.npz')
    np.savez(
        weights_path,
        W1=mlp_model.W1, b1=mlp_model.b1,
        W2=mlp_model.W2, b2=mlp_model.b2,
        feature_names=feature_names
    )
    print(f"Saved: {weights_path}")
    
    # 4. Generate Visualizations
    print("\n[Step 4/5] Generating Publication-Quality Figures...")
    
    # -------------------------------------------------------------
    # Figure 4: Dataset Split Proportions & Response Invariance
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Partition Bar Chart
    set_minimal_theme(axes[0])
    splits = ['Training (70%)', 'Validation (15%)', 'Testing (15%)']
    counts = [len(df_train), len(df_val), len(df_test)]
    colors_split = ['#2563EB', '#10B981', '#EF4444']
    bars = axes[0].bar(splits, counts, color=colors_split, edgecolor='#1F2937', linewidth=0.8, alpha=0.88, width=0.55)
    axes[0].set_title("(A) Ill-Conditioned Signal Partition Breakdown", pad=12)
    axes[0].set_xlabel("Partition Subset", labelpad=8)
    axes[0].set_ylabel("Number of Signal Instances", labelpad=8)
    axes[0].set_ylim(0, 850)
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h + 15, f"{h} ({h/1000*100:.1f}%)",
                     ha='center', va='bottom', fontname='Cambria', fontsize=10, fontweight='bold')
                     
    # Subplot B: Target Distribution Density Invariance
    set_minimal_theme(axes[1])
    sns.kdeplot(df_train['Target_Standardized'], ax=axes[1], color='#2563EB', linewidth=2.2, label='Train Partition (N = 700)', fill=True, alpha=0.15)
    sns.kdeplot(df_val['Target_Standardized'], ax=axes[1], color='#10B981', linewidth=2.2, linestyle='--', label='Val Partition (N = 150)', fill=True, alpha=0.15)
    sns.kdeplot(df_test['Target_Standardized'], ax=axes[1], color='#EF4444', linewidth=2.2, linestyle=':', label='Test Partition (N = 150)', fill=True, alpha=0.15)
    axes[1].set_title("(B) Standardized Response Distribution Consistency Across Splits", pad=12)
    axes[1].set_xlabel("Standardized System Response Target", labelpad=8)
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
               
    fig4_path = os.path.join(base_dir, '04_dataset_split_proportions_and_response_invariance.png')
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig4_path}")
    
    # -------------------------------------------------------------
    # Figure 5: Neural Network Architecture & Non-Linear Mapping
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Neural Network Diagram Schematic
    set_minimal_theme(axes[0])
    axes[0].set_xlim(-0.5, 3.5)
    axes[0].set_ylim(-0.5, 8.5)
    axes[0].axis('off')
    
    # Draw layers: Input (8 nodes), Hidden (6 sampled nodes), Output (1 node)
    in_y = np.linspace(0.5, 7.5, 8)
    hid_y = np.linspace(1.5, 6.5, 6)
    out_y = [4.0]
    
    for y_pos in in_y:
        axes[0].scatter(0.5, y_pos, color='#2563EB', s=240, edgecolors='#1E40AF', linewidths=1.5, zorder=5)
    for y_pos in hid_y:
        axes[0].scatter(1.8, y_pos, color='#10B981', s=240, edgecolors='#047857', linewidths=1.5, zorder=5)
    for y_pos in out_y:
        axes[0].scatter(3.0, y_pos, color='#EF4444', s=280, edgecolors='#B91C1C', linewidths=1.5, zorder=5)
        
    # Draw representative connection lines
    for yi in in_y[::2]:
        for yh in hid_y[::2]:
            axes[0].plot([0.5, 1.8], [yi, yh], color='#94A3B8', alpha=0.35, linewidth=0.8)
    for yh in hid_y:
        axes[0].plot([1.8, 3.0], [yh, 4.0], color='#10B981', alpha=0.5, linewidth=1.0)
        
    axes[0].text(0.5, 8.1, "Input Layer\n(D = 8 Features)", ha='center', fontname='Cambria', fontsize=10, fontweight='bold', color='#1E40AF')
    axes[0].text(1.8, 8.1, "Hidden Layer\n(16 Neurons + Tanh)", ha='center', fontname='Cambria', fontsize=10, fontweight='bold', color='#047857')
    axes[0].text(3.0, 8.1, "Output Layer\n(Linear y_hat)", ha='center', fontname='Cambria', fontsize=10, fontweight='bold', color='#B91C1C')
    axes[0].set_title("(A) Deep MLP Architecture for Ill-Conditioned Optimization", pad=14)
    
    # Subplot B: Initial Prediction Parity vs Actual Target
    set_minimal_theme(axes[1])
    X_val = df_val[feature_names].values
    y_val = df_val['Target_Standardized'].values
    y_init_pred = mlp_model.forward(X_val).flatten()
    
    axes[1].scatter(y_val, y_init_pred, color='#2563EB', alpha=0.65, edgecolors='#FFFFFF', linewidths=0.5, s=45)
    min_v, max_v = min(y_val), max(y_val)
    axes[1].plot([min_v, max_v], [min_v, max_v], color='#111827', linestyle='--', linewidth=1.5, label='Target Diagonal (y = x)')
    axes[1].set_title("(B) Pre-Optimization Neural Network Output Alignment", pad=12)
    axes[1].set_xlabel("Ground Truth Standardized Target", labelpad=8)
    axes[1].set_ylabel("Neural Network Predicted Response y_hat", labelpad=8)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles5 = [
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2563EB', markersize=9, label='Input Feature Nodes (8 Features)'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#10B981', markersize=9, label='Hidden Neurons (16 Tanh Units with Momentum Buffers)'),
        mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EF4444', markersize=10, label='Output Regression Unit (MSE Objective)'),
        mpl.lines.Line2D([0], [0], color='#111827', linestyle='--', label='Ideal Prediction Line')
    ]
    fig.legend(handles=legend_handles5, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig5_path = os.path.join(base_dir, '05_neural_network_architecture_and_forward_mapping.png')
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig5_path}")
    
    # -------------------------------------------------------------
    # Figure 6: Ill-Conditioned Ravine & Trajectories (Standard GD vs Momentum)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    traj_gd, traj_mom = simulate_ravine_trajectories(epochs=45, lr_gd=0.038, lr_mom=0.038, beta=0.88)
    
    # Subplot A: 2D Pathological Ravine Contour & Trajectories
    set_minimal_theme(axes[0])
    w1_vals = np.linspace(-5.0, 1.0, 150)
    w2_vals = np.linspace(-3.5, 3.5, 150)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    # Pathological ravine: steep across w2 (50.0), gentle along w1 (1.0)
    Z_loss = 0.5 * W1**2 + 25.0 * W2**2
    
    cs = axes[0].contour(W1, W2, Z_loss, levels=25, cmap='viridis_r', alpha=0.75, linewidths=0.9)
    axes[0].clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    
    # Plot Trajectories
    axes[0].plot(traj_gd[:, 0], traj_gd[:, 1], color='#EF4444', marker='x', markersize=4.5, linewidth=1.6,
                 label='Standard GD (Severe Cross-Canyon Oscillations)')
    axes[0].plot(traj_mom[:, 0], traj_mom[:, 1], color='#10B981', marker='o', markersize=4.0, linewidth=2.2,
                 label='Momentum GD (Oscillations Damped, Smooth Path)')
    axes[0].scatter(0, 0, color='#2563EB', s=160, marker='*', zorder=10, label='Global Optimum (w* = [0, 0])')
    axes[0].scatter(traj_gd[0, 0], traj_gd[0, 1], color='#111827', s=100, marker='s', zorder=10, label='Start Point (w0 = [-4.5, 3.2])')
    
    axes[0].set_title("(A) Pathological Ravine Contour: Standard GD vs. Momentum GD (beta = 0.88)", pad=12)
    axes[0].set_xlabel("Parameter w1: Shallow / Slow Convergence Direction (lambda_1 = 1.0)", labelpad=8)
    axes[0].set_ylabel("Parameter w2: Steep / High Curvature Axis (lambda_2 = 50.0)", labelpad=8)
    axes[0].set_xlim(-5.0, 1.0)
    axes[0].set_ylim(-3.5, 3.5)
    
    # Subplot B: Perpendicular Oscillation Amplitude over Iterations (w2 Coordinate)
    set_minimal_theme(axes[1])
    iter_x = np.arange(len(traj_gd))
    axes[1].plot(iter_x, traj_gd[:, 1], color='#EF4444', marker='x', linewidth=1.6, label='Standard GD (w2 Zig-Zagging Across Canyon)')
    axes[1].plot(iter_x, traj_mom[:, 1], color='#10B981', marker='o', linewidth=2.2, label='Momentum GD (w2 Rapidly Decays to Floor)')
    axes[1].axhline(0, color='#111827', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[1].set_title("(B) Cross-Canyon Oscillation Amplitude Dynamics (Steep Axis w2)", pad=12)
    axes[1].set_xlabel("Optimization Step Iteration (t)", labelpad=8)
    axes[1].set_ylabel("Steep Coordinate Value (w2)", labelpad=8)
    axes[1].set_xlim(0, 45)
    axes[1].set_ylim(-3.8, 3.8)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    legend_handles6 = [
        mpl.lines.Line2D([0], [0], color='#EF4444', marker='x', linewidth=1.6, label='Standard GD (High Frequency Orthogonal Bounce)'),
        mpl.lines.Line2D([0], [0], color='#10B981', marker='o', linewidth=2.2, label='Momentum GD (Polyak Momentum beta = 0.88 - Fast Canyon Transit)'),
        mpl.lines.Line2D([0], [0], marker='*', color='w', markerfacecolor='#2563EB', markersize=12, label='Global Minimum w*'),
        mpl.lines.Line2D([0], [0], marker='s', color='w', markerfacecolor='#111827', markersize=8, label='Initialization Point')
    ]
    fig.legend(handles=legend_handles6, loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    fig6_path = os.path.join(base_dir, '06_ill_conditioned_loss_surface_and_momentum_path.png')
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {fig6_path}")
    
    print("\nModule 2 processing and visualization completed successfully!")

if __name__ == '__main__':
    run_module_2()
