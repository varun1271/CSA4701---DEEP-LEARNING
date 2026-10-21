"""
Module 3: Training & Evaluation
Scenario: A retail enterprise collects customer data (Income, Spending, Purchase Frequency).
Analyze the covariance matrix and determine which features show strong positive or negative relationships.
Model Probability Distributions and Bayes' Theorem.

This script:
1. Loads partitioned datasets (Train, Val, Test) and serialized model parameters from Module 2.
2. Benchmarks Computational Performance:
   - Method A: Sequential Scalar Loop Bayesian Inference.
   - Method B: Vectorized Matrix Bayesian Inference (NumPy batch quadratic forms).
   - Evaluates latency across customer batch scales (N = 100 to 10,000).
3. Evaluates Bayes' Theorem Model on Unseen Held-Out Test Split (152 customers):
   - Accuracy, Precision, Recall, Specificity, F1-Score.
   - ROC-AUC (Receiver Operating Characteristic) and PR-AUC.
   - Log-Loss / Cross-Entropy and Brier Score (probability calibration).
   - Confusion Matrix (True Positives, False Positives, True Negatives, False Negatives).
4. Conducts Comprehensive Covariance and Correlation Strength Analysis:
   - Quantifies and categorizes strong positive, strong negative, and orthogonal feature pairs.
5. Exports benchmark results, metrics summaries, covariance analysis, and sample predictions to CSV.
6. Generates publication-grade, colorful visualization graphs:
   - Font: Cambria
   - Minimalist theme with light grid
   - Clear X and Y labels with well-numbered scales
   - Legend positioned cleanly at the bottom
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
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

def bayesian_predict_sequential_loop(X, prior_c0, prior_c1, mu_c0, mu_c1, cov_c0, cov_c1):
    """
    Sequential scalar loop implementation of Bayesian inference.
    Processes each customer record individually.
    """
    n_samples = len(X)
    posteriors = np.zeros(n_samples)
    mvn_c0 = stats.multivariate_normal(mu_c0, cov_c0, allow_singular=True)
    mvn_c1 = stats.multivariate_normal(mu_c1, cov_c1, allow_singular=True)
    
    for i in range(n_samples):
        x_i = X[i]
        lik_0 = mvn_c0.pdf(x_i)
        lik_1 = mvn_c1.pdf(x_i)
        evid = lik_0 * prior_c0 + lik_1 * prior_c1
        if evid > 0:
            posteriors[i] = (lik_1 * prior_c1) / evid
        else:
            posteriors[i] = prior_c1
    return posteriors

def bayesian_predict_vectorized_matrix(X, prior_c0, prior_c1, mu_c0, mu_c1, cov_c0, cov_c1):
    """
    Vectorized matrix implementation of Bayesian inference using BLAS and quadratic forms.
    Computes all samples simultaneously in matrix operations.
    """
    D = X.shape[1]
    inv_c0 = np.linalg.pinv(cov_c0)
    inv_c1 = np.linalg.pinv(cov_c1)
    
    sign0, logdet0 = np.linalg.slogdet(cov_c0)
    sign1, logdet1 = np.linalg.slogdet(cov_c1)
    
    # Quadratic Mahalanobis forms in batch
    diff_0 = X - mu_c0
    diff_1 = X - mu_c1
    
    maha_0 = np.sum((diff_0 @ inv_c0) * diff_0, axis=1)
    maha_1 = np.sum((diff_1 @ inv_c1) * diff_1, axis=1)
    
    log_lik_0 = -0.5 * (D * np.log(2 * np.pi) + logdet0 + maha_0)
    log_lik_1 = -0.5 * (D * np.log(2 * np.pi) + logdet1 + maha_1)
    
    log_joint_0 = log_lik_0 + np.log(prior_c0)
    log_joint_1 = log_lik_1 + np.log(prior_c1)
    
    # Log-sum-exp trick for numerical stability
    max_log = np.maximum(log_joint_0, log_joint_1)
    log_evidence = max_log + np.log(np.exp(log_joint_0 - max_log) + np.exp(log_joint_1 - max_log))
    
    posterior_c1 = np.exp(log_joint_1 - log_evidence)
    return posterior_c1

def run_module_3():
    module3_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(module3_dir)
    module2_dir = os.path.join(base_dir, 'Data Splitting & Model Design')
    
    print("=" * 75)
    print("MODULE 3: TRAINING & EVALUATION (RETAIL CUSTOMER ANALYTICS)")
    print("=" * 75)
    
    train_csv_path = os.path.join(module2_dir, 'train_customers.csv')
    val_csv_path = os.path.join(module2_dir, 'val_customers.csv')
    test_csv_path = os.path.join(module2_dir, 'test_customers.csv')
    params_path = os.path.join(module2_dir, 'class_priors_and_params.npz')
    
    if not os.path.exists(test_csv_path) or not os.path.exists(params_path):
        raise FileNotFoundError("Module 2 output files missing. Please run Module 2 first.")
        
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)
    
    params = np.load(params_path, allow_pickle=True)
    prior_c0 = float(params['prior_c0'])
    prior_c1 = float(params['prior_c1'])
    mu_c0 = params['mu_c0']
    mu_c1 = params['mu_c1']
    cov_c0 = params['cov_c0']
    cov_c1 = params['cov_c1']
    feature_cols = list(params['feature_names'])
    
    X_train = df_train[feature_cols].values
    y_train = df_train['High_Value_Customer'].values
    
    X_val = df_val[feature_cols].values
    y_val = df_val['High_Value_Customer'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['High_Value_Customer'].values
    
    # 1. Computational Benchmark: Loops vs. Vectorized Matrix Inference
    print("\nExecuting Computational Benchmark (Sequential Loops vs. Vectorized Matrix)...")
    scales = [100, 250, 500, 1000, 2500, 5000, 10000]
    benchmark_results = []
    
    np.random.seed(42)
    for N in scales:
        # Sample or replicate customers to reach scale N
        sample_indices = np.random.choice(len(X_train), size=N, replace=True)
        X_bench = X_train[sample_indices]
        
        # Benchmark Method A: Sequential Loop
        t0 = time.perf_counter()
        _ = bayesian_predict_sequential_loop(X_bench, prior_c0, prior_c1, mu_c0, mu_c1, cov_c0, cov_c1)
        t_loop = (time.perf_counter() - t0) * 1000.0  # ms
        
        # Benchmark Method B: Vectorized Matrix
        t0 = time.perf_counter()
        _ = bayesian_predict_vectorized_matrix(X_bench, prior_c0, prior_c1, mu_c0, mu_c1, cov_c0, cov_c1)
        t_matrix = (time.perf_counter() - t0) * 1000.0  # ms
        
        speedup = t_loop / max(t_matrix, 1e-4)
        benchmark_results.append({
            'Scale_N': N,
            'Sequential_Loop_ms': round(t_loop, 2),
            'Vectorized_Matrix_ms': round(t_matrix, 3),
            'Speedup_Factor': round(speedup, 1)
        })
        print(f"  - Scale N = {N:5d} | Loops: {t_loop:8.2f} ms | Matrix: {t_matrix:6.3f} ms | Speedup: {speedup:6.1f}x")
        
    df_bench = pd.DataFrame(benchmark_results)
    bench_csv_path = os.path.join(module3_dir, 'computational_benchmark_timing.csv')
    df_bench.to_csv(bench_csv_path, index=False)
    
    # 2. Validation Threshold Calibration
    print("\nCalibrating Decision Threshold on Validation Set...")
    val_probs = bayesian_predict_vectorized_matrix(X_val, prior_c0, prior_c1, mu_c0, mu_c1, cov_c0, cov_c1)
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thresh = 0.5
    best_f1 = 0.0
    for th in thresholds:
        f1 = f1_score(y_val, (val_probs >= th).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
    print(f"  * Optimal Validation Threshold: tau = {best_thresh:.2f} (F1 = {best_f1:.4f})")
    
    # 3. Model Evaluation on Unseen Test Split
    print("\nEvaluating Bayesian Classifier on Unseen Held-Out Test Set (152 customers)...")
    test_probs = bayesian_predict_vectorized_matrix(X_test, prior_c0, prior_c1, mu_c0, mu_c1, cov_c0, cov_c1)
    test_preds = (test_probs >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    ll = log_loss(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)
    
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    spec = tn / (tn + fp)
    
    print("-" * 75)
    print("TEST SET EVALUATION METRICS:")
    print("-" * 75)
    print(f"  * Accuracy:             {acc:.4f} ({acc*100:.2f}%)")
    print(f"  * Precision:            {prec:.4f} ({prec*100:.2f}%)")
    print(f"  * Recall (Sensitivity): {rec:.4f} ({rec*100:.2f}%)")
    print(f"  * Specificity:          {spec:.4f} ({spec*100:.2f}%)")
    print(f"  * F1-Score:             {f1:.4f}")
    print(f"  * ROC-AUC Score:        {roc_auc:.4f}")
    print(f"  * PR-AUC (Avg Prec):    {pr_auc:.4f}")
    print(f"  * Log-Loss:             {ll:.4f}")
    print(f"  * Brier Score:          {brier:.4f}")
    print(f"  * Confusion Matrix:     TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    metrics_summary = pd.DataFrame([{
        'Metric': 'Accuracy', 'Value': acc, 'Interpretation': 'Overall fraction of correct predictions'
    }, {
        'Metric': 'Precision', 'Value': prec, 'Interpretation': 'Fraction of flagged VIPs who are truly VIP'
    }, {
        'Metric': 'Recall', 'Value': rec, 'Interpretation': 'Proportion of true VIPs successfully identified'
    }, {
        'Metric': 'Specificity', 'Value': spec, 'Interpretation': 'Proportion of standard shoppers correctly identified'
    }, {
        'Metric': 'F1-Score', 'Value': f1, 'Interpretation': 'Harmonic mean of Precision and Recall'
    }, {
        'Metric': 'ROC-AUC', 'Value': roc_auc, 'Interpretation': 'Area Under Receiver Operating Characteristic Curve'
    }, {
        'Metric': 'PR-AUC', 'Value': pr_auc, 'Interpretation': 'Area Under Precision-Recall Curve'
    }, {
        'Metric': 'Log-Loss', 'Value': ll, 'Interpretation': 'Negative log-likelihood cross-entropy loss'
    }, {
        'Metric': 'Brier Score', 'Value': brier, 'Interpretation': 'Mean squared probability calibration error'
    }])
    metrics_csv_path = os.path.join(module3_dir, 'bayesian_classification_metrics_summary.csv')
    metrics_summary.to_csv(metrics_csv_path, index=False)
    
    # 4. Detailed Covariance & Correlation Strength Analysis Table
    print("\nCompiling Detailed Covariance & Correlation Findings...")
    cov_test = np.cov(X_test, rowvar=False)
    corr_test = np.corrcoef(X_test, rowvar=False)
    
    cov_findings = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            f1_name, f2_name = feature_cols[i], feature_cols[j]
            r = corr_test[i, j]
            cov = cov_test[i, j]
            
            if r >= 0.70:
                cat = "Strong Positive"
                interp = "Direct revenue multiplier; features scale strongly in tandem."
            elif r >= 0.40:
                cat = "Moderate Positive"
                interp = "Meaningful mutual synergy; higher values generally align."
            elif r <= -0.60:
                cat = "Strong Negative"
                interp = "Inverse dependency; high values sharply decrease the other."
            elif r <= -0.35:
                cat = "Moderate Negative"
                interp = "Moderate trade-off or inverse customer behavior."
            else:
                cat = "Near-Zero / Orthogonal"
                interp = "Independent behavioral dimensions; no direct linear association."
                
            cov_findings.append({
                'Feature_1': f1_name,
                'Feature_2': f2_name,
                'Covariance': round(cov, 2),
                'Pearson_r': round(r, 3),
                'Relationship_Category': cat,
                'Enterprise_Interpretation': interp
            })
            
    df_cov_findings = pd.DataFrame(cov_findings)
    df_cov_findings.sort_values(by='Pearson_r', ascending=False, inplace=True)
    cov_csv_path = os.path.join(module3_dir, 'covariance_correlation_analysis_summary.csv')
    df_cov_findings.to_csv(cov_csv_path, index=False)
    
    # Export sample test predictions
    df_test_export = df_test.copy()
    df_test_export['Posterior_VIP_Prob'] = np.round(test_probs, 4)
    df_test_export['Predicted_VIP'] = test_preds
    preds_csv_path = os.path.join(module3_dir, 'test_customer_bayesian_predictions.csv')
    df_test_export.to_csv(preds_csv_path, index=False)
    
    # 5. Publication-Grade Visualization Graphs
    print("\nGenerating Module 3 Visualization Graphs...")
    
    # Graph 7: ROC, Precision-Recall, and Probability Calibration Curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Panel A: ROC Curve
    set_minimal_theme(axes[0])
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    axes[0].plot(fpr, tpr, color='#2563EB', linewidth=2.4, label=f'Bayesian Classifier (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='#9CA3AF', linestyle='--', linewidth=1.5, label='Random Chance (AUC = 0.50)')
    axes[0].set_title("(A) Receiver Operating Characteristic (ROC)", pad=12)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", labelpad=8)
    axes[0].set_ylabel("True Positive Rate (Recall)", labelpad=8)
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    
    # Panel B: Precision-Recall Curve
    set_minimal_theme(axes[1])
    prec_pts, rec_pts, _ = precision_recall_curve(y_test, test_probs)
    axes[1].plot(rec_pts, prec_pts, color='#EC4899', linewidth=2.4, label=f'Precision-Recall (PR-AUC = {pr_auc:.3f})')
    axes[1].axhline(np.mean(y_test), color='#9CA3AF', linestyle='--', linewidth=1.5, label=f'Baseline Rate ({np.mean(y_test):.2f})')
    axes[1].set_title("(B) Precision-Recall Curve", pad=12)
    axes[1].set_xlabel("Recall (Coverage of True VIPs)", labelpad=8)
    axes[1].set_ylabel("Precision (Accuracy of VIP Calls)", labelpad=8)
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    
    # Panel C: Calibration Curve (Reliability Diagram)
    set_minimal_theme(axes[2])
    prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=8, strategy='uniform')
    axes[2].plot(prob_pred, prob_true, marker='o', markersize=6, color='#10B981', linewidth=2.2, label=f'Bayesian Posterior (Brier = {brier:.3f})')
    axes[2].plot([0, 1], [0, 1], color='#9CA3AF', linestyle='--', linewidth=1.5, label='Perfect Calibration')
    axes[2].set_title("(C) Reliability Diagram (Probability Calibration)", pad=12)
    axes[2].set_xlabel("Predicted Posterior Probability", labelpad=8)
    axes[2].set_ylabel("Empirical True Fraction", labelpad=8)
    axes[2].set_xlim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.02)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    handles3, labels3 = axes[2].get_legend_handles_labels()
    all_handles = [handles1[0], handles2[0], handles3[0], handles3[1]]
    all_labels = [labels1[0], labels2[0], labels3[0], labels3[1]]
    fig.legend(handles=all_handles, labels=all_labels, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph7_path = os.path.join(module3_dir, '07_roc_pr_and_calibration_curves.png')
    plt.savefig(graph7_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph7_path}")
    
    # Graph 8: Computational Speedup Benchmark
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Latency Comparison (Log Scale)
    set_minimal_theme(axes[0])
    axes[0].plot(df_bench['Scale_N'], df_bench['Sequential_Loop_ms'], marker='s', markersize=6, color='#EF4444', 
                 linewidth=2.2, label='Sequential Loops (Scalar Python)')
    axes[0].plot(df_bench['Scale_N'], df_bench['Vectorized_Matrix_ms'], marker='o', markersize=6, color='#10B981', 
                 linewidth=2.2, label='Vectorized Matrix (NumPy BLAS)')
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].set_title("(A) Execution Latency Benchmark vs. Batch Size", pad=12)
    axes[0].set_xlabel("Customer Batch Size (N)", labelpad=8)
    axes[0].set_ylabel("Execution Latency (Milliseconds, Log Scale)", labelpad=8)
    
    # Subplot B: Speedup Factor Bar Chart
    set_minimal_theme(axes[1])
    bar_pos = np.arange(len(df_bench))
    bar_colors = ['#3B82F6', '#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#06B6D4']
    bars = axes[1].bar(bar_pos, df_bench['Speedup_Factor'], color=bar_colors, edgecolor='#1F2937', linewidth=0.8, alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, height + 8, f"{height:.0f}x", 
                     ha='center', va='bottom', fontname='Cambria', fontsize=9.5, fontweight='bold', color='#1F2937')
    axes[1].set_xticks(bar_pos)
    axes[1].set_xticklabels([f"N={n}" for n in df_bench['Scale_N']])
    axes[1].set_title("(B) Vectorized Matrix Speedup Acceleration Factor", pad=12)
    axes[1].set_xlabel("Customer Batch Size (N)", labelpad=8)
    axes[1].set_ylabel("Speedup Multiplier Factor (X-Times Faster)", labelpad=8)
    axes[1].set_ylim(0, max(df_bench['Speedup_Factor']) * 1.18)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    handles_b, labels_b = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles_b, labels=labels_b, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=2, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph8_path = os.path.join(module3_dir, '08_bayesian_computational_benchmark.png')
    plt.savefig(graph8_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph8_path}")
    
    # Graph 9: Confusion Matrix & Posterior Confidence Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Normalized Confusion Matrix
    set_minimal_theme(axes[0])
    cm = confusion_matrix(y_test, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    labels_cm = ['Standard (C_0)', 'High-Value (C_1)']
    sns.heatmap(cm_norm, ax=axes[0], annot=True, fmt=".2%", cmap='Blues', cbar=False,
                xticklabels=labels_cm, yticklabels=labels_cm, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title(f"(A) Normalized Confusion Matrix (Test Accuracy: {acc*100:.1f}%)", pad=12)
    axes[0].set_xlabel("Predicted Customer Tier", labelpad=8)
    axes[0].set_ylabel("Ground-Truth Customer Tier", labelpad=8)
    
    # Subplot B: Posterior Probability Distribution (Confidence Separation)
    set_minimal_theme(axes[1])
    probs_c0 = test_probs[y_test == 0]
    probs_c1 = test_probs[y_test == 1]
    
    sns.histplot(probs_c0, ax=axes[1], color='#3B82F6', edgecolor='#1D4ED8', alpha=0.6, bins=20, label='True Standard (C_0)', stat='density')
    sns.histplot(probs_c1, ax=axes[1], color='#EF4444', edgecolor='#B91C1C', alpha=0.6, bins=20, label='True High-Value (C_1)', stat='density')
    axes[1].axvline(best_thresh, color='#1F2937', linestyle='--', linewidth=2.0, label=f'Decision Cutoff ({best_thresh:.2f})')
    axes[1].set_title("(B) Bayesian Posterior Probability Confidence Separation", pad=12)
    axes[1].set_xlabel("Predicted Posterior Probability P(High-Value | x)", labelpad=8)
    axes[1].set_ylabel("Sample Probability Density", labelpad=8)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    handles9, labels9 = axes[1].get_legend_handles_labels()
    fig.legend(handles=handles9, labels=labels9, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=3, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph9_path = os.path.join(module3_dir, '09_confusion_matrix_and_posterior_confidence.png')
    plt.savefig(graph9_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph9_path}")
    
    print("\nModule 3 execution completed successfully!")

if __name__ == '__main__':
    run_module_3()
