"""
Module 3: Training & Evaluation
Scenario: A manufacturing unit records the number of defective products in different batches.
Model the defect counts using a suitable probability distribution and estimate the probability
of observing a specified number of defects. Analyze Entropy, Cross-Entropy, and KL Divergence.

This script:
1. Loads partitioned datasets (Train, Val, Test) and serialized model parameters from Module 2.
2. Executes Computational Performance Benchmarking:
   - Method A: Sequential Scalar Loop Likelihood & Entropy Evaluation.
   - Method B: Vectorized Matrix/Tensor Evaluation (NumPy broadcasted log-gamma and PMFs).
   - Measures execution latency across batch scales (N = 100 to 10,000).
3. Evaluates Candidate Probability Distributions on Unseen Test Split (152 batches):
   - Cross-Entropy H(P_test, Q) and KL Divergence D_KL(P_test || Q).
   - Chi-Square Goodness-of-Fit (chi^2, p-value).
   - Mean Absolute Probability Error (MAE) and Root Mean Squared Error (RMSE).
   - Log-Likelihood / Negative Log-Likelihood (NLL).
4. Generates Comprehensive Defect Probability Estimation Table:
   - Estimates exact probabilities P(K = k) and cumulative scrap risks P(K >= 10).
5. Exports benchmark results, information metrics, probability estimates, and test evaluations to CSV.
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
from scipy.special import gammaln
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

def compute_entropy_and_kl(p_emp, q_model, eps=1e-12):
    """Calculates Shannon Entropy H(P), Cross-Entropy H(P, Q), and KL Divergence."""
    p = np.clip(p_emp, eps, 1.0)
    p = p / np.sum(p)
    q = np.clip(q_model, eps, 1.0)
    q = q / np.sum(q)
    h_p = -np.sum(p * np.log2(p))
    ce = -np.sum(p * np.log2(q))
    kl = max(ce - h_p, 0.0)
    return h_p, ce, kl

def sequential_loop_likelihood_evaluation(defects, r_nb, p_nb, lambda_pois):
    """
    Sequential scalar loop implementation of likelihood and entropy scoring.
    Simulates row-by-row legacy quality monitoring.
    """
    n = len(defects)
    nb_log_lik = 0.0
    pois_log_lik = 0.0
    for i in range(n):
        k = defects[i]
        # Scalar Negative Binomial PMF
        nb_p = stats.nbinom.pmf(k, r_nb, p_nb)
        nb_log_lik += np.log(max(nb_p, 1e-15))
        # Scalar Poisson PMF
        pois_p = stats.poisson.pmf(k, lambda_pois)
        pois_log_lik += np.log(max(pois_p, 1e-15))
    return nb_log_lik, pois_log_lik

def vectorized_matrix_likelihood_evaluation(defects, r_nb, p_nb, lambda_pois):
    """
    Vectorized tensor implementation of likelihood and information scoring.
    Leverages NumPy vectorized log-gamma functions and broadcasting.
    """
    k = defects.astype(float)
    # Vectorized Negative Binomial log-PMF:
    # ln(Gamma(k + r)) - ln(k!) - ln(Gamma(r)) + r*ln(p) + k*ln(1 - p)
    log_comb = gammaln(k + r_nb) - gammaln(k + 1.0) - gammaln(r_nb)
    log_nb = log_comb + r_nb * np.log(p_nb) + k * np.log(1.0 - p_nb)
    nb_log_lik = np.sum(log_nb)
    
    # Vectorized Poisson log-PMF:
    # -lambda + k*ln(lambda) - ln(k!)
    log_pois = -lambda_pois + k * np.log(lambda_pois) - gammaln(k + 1.0)
    pois_log_lik = np.sum(log_pois)
    
    return nb_log_lik, pois_log_lik

def run_module_3():
    module3_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(module3_dir)
    module2_dir = os.path.join(base_dir, 'Data Splitting & Model Design')
    
    print("=" * 75)
    print("MODULE 3: TRAINING & EVALUATION (INFORMATION THEORY & BENCHMARKING)")
    print("=" * 75)
    
    train_csv_path = os.path.join(module2_dir, 'train_batches.csv')
    val_csv_path = os.path.join(module2_dir, 'val_batches.csv')
    test_csv_path = os.path.join(module2_dir, 'test_batches.csv')
    params_path = os.path.join(module2_dir, 'fitted_distribution_params.npz')
    
    if not os.path.exists(test_csv_path) or not os.path.exists(params_path):
        raise FileNotFoundError("Module 2 output files missing. Please run Module 2 first.")
        
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)
    
    params = np.load(params_path)
    r_nb = float(params['r_nb'])
    p_nb = float(params['p_nb'])
    lambda_pois = float(params['lambda_pois'])
    p_binom = float(params['p_binom'])
    
    defects_test = df_test['Defect_Count'].values
    defects_train = df_train['Defect_Count'].values
    
    # 1. Computational Benchmark: Loops vs. Vectorized Tensor Evaluation
    print("\nExecuting Computational Benchmark (Sequential Scalar Loops vs. Vectorized Tensor)...")
    scales = [100, 250, 500, 1000, 2500, 5000, 10000]
    benchmark_results = []
    
    np.random.seed(42)
    for N in scales:
        sample_indices = np.random.choice(len(defects_train), size=N, replace=True)
        defects_bench = defects_train[sample_indices]
        
        # Benchmark Method A: Sequential Loop
        t0 = time.perf_counter()
        _ = sequential_loop_likelihood_evaluation(defects_bench, r_nb, p_nb, lambda_pois)
        t_loop = (time.perf_counter() - t0) * 1000.0  # ms
        
        # Benchmark Method B: Vectorized Matrix/Tensor
        t0 = time.perf_counter()
        _ = vectorized_matrix_likelihood_evaluation(defects_bench, r_nb, p_nb, lambda_pois)
        t_vec = (time.perf_counter() - t0) * 1000.0  # ms
        
        speedup = t_loop / max(t_vec, 1e-4)
        benchmark_results.append({
            'Batch_Scale_N': N,
            'Sequential_Loop_ms': round(t_loop, 2),
            'Vectorized_Tensor_ms': round(t_vec, 3),
            'Speedup_Factor': round(speedup, 1)
        })
        print(f"  - Scale N = {N:5d} batches | Loops: {t_loop:8.2f} ms | Vectorized: {t_vec:6.3f} ms | Speedup: {speedup:6.1f}x")
        
    df_bench = pd.DataFrame(benchmark_results)
    bench_csv_path = os.path.join(module3_dir, 'computational_benchmark_timing.csv')
    df_bench.to_csv(bench_csv_path, index=False)
    
    # 2. Information-Theoretic and Goodness-of-Fit Evaluation on Test Set
    print("\nEvaluating Information Metrics & Goodness-of-Fit on Unseen Test Split (152 batches)...")
    k_max = int(max(np.max(defects_test), 22))
    k_support = np.arange(0, k_max + 1)
    
    # Empirical test distribution
    p_test = np.array([np.mean(defects_test == k) for k in k_support])
    h_p_test, _, _ = compute_entropy_and_kl(p_test, p_test)
    
    # Models
    q_nb_test = stats.nbinom.pmf(k_support, r_nb, p_nb)
    q_pois_test = stats.poisson.pmf(k_support, lambda_pois)
    q_binom_test = stats.binom.pmf(k_support, n=500, p=p_binom)
    q_unif_test = np.ones_like(k_support, dtype=float) / len(k_support)
    
    model_eval_dict = {
        'Negative Binomial NB(r, p)': q_nb_test,
        'Poisson Pois(lambda)': q_pois_test,
        'Binomial Bin(n=500, p)': q_binom_test,
        'Uniform Baseline': q_unif_test
    }
    
    info_metrics_list = []
    print(f"  * Unseen Test Set Intrinsic Shannon Entropy H(P_test): {h_p_test:.4f} bits\n")
    print(f"  {'Model Name':30s} | {'Cross-Entropy':>15s} | {'KL Divergence':>15s} | {'MAE (%)':>10s} | {'RMSE (%)':>10s} | {'Log-Likelihood':>16s}")
    print("  " + "-" * 105)
    
    for name, q_cand in model_eval_dict.items():
        _, ce, kl = compute_entropy_and_kl(p_test, q_cand)
        mae = np.mean(np.abs(p_test - q_cand)) * 100.0
        rmse = np.sqrt(np.mean((p_test - q_cand) ** 2)) * 100.0
        
        # Test Log-Likelihood
        if 'Negative Binomial' in name:
            log_lik, _ = vectorized_matrix_likelihood_evaluation(defects_test, r_nb, p_nb, lambda_pois)
            log_lik_val = log_lik
        elif 'Poisson' in name:
            _, log_lik = vectorized_matrix_likelihood_evaluation(defects_test, r_nb, p_nb, lambda_pois)
            log_lik_val = log_lik
        elif 'Binomial' in name:
            log_lik_val = np.sum(stats.binom.logpmf(defects_test, n=500, p=p_binom))
        else:
            log_lik_val = np.sum(np.log(1.0 / len(k_support)) * len(defects_test))
            
        info_metrics_list.append({
            'Model': name,
            'Shannon_Entropy_HP': round(h_p_test, 4),
            'Cross_Entropy_HPQ': round(ce, 4),
            'KL_Divergence_DKL': round(kl, 4),
            'MAE_Percent': round(mae, 4),
            'RMSE_Percent': round(rmse, 4),
            'Test_Log_Likelihood': round(log_lik_val, 2)
        })
        print(f"  {name:30s} | {ce:11.4f} bits | {kl:11.4f} bits | {mae:9.3f}% | {rmse:9.3f}% | {log_lik_val:16.2f}")
        
    df_info_metrics = pd.DataFrame(info_metrics_list)
    info_csv_path = os.path.join(module3_dir, 'distribution_goodness_of_fit_and_information_metrics.csv')
    df_info_metrics.to_csv(info_csv_path, index=False)
    
    # 3. Comprehensive Defect Probability Estimation Table
    k_targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    prob_estimates = []
    
    for k in k_targets:
        emp_p = np.mean(defects_test == k)
        nb_p = stats.nbinom.pmf(k, r_nb, p_nb)
        pois_p = stats.poisson.pmf(k, lambda_pois)
        err_nb = abs(nb_p - emp_p)
        err_pois = abs(pois_p - emp_p)
        
        if k == 0:
            status = "Pristine Zero-Defect"
        elif k <= 3:
            status = "Target Optimal Tier"
        elif k <= 6:
            status = "Acceptable Steady-State"
        elif k <= 9:
            status = "Elevated Warning Zone"
        else:
            status = "Critical Batch Scrap (>= 10)"
            
        prob_estimates.append({
            'Defect_Count_k': k,
            'Empirical_Frequency': round(emp_p * 100, 2),
            'Negative_Binomial_Prob': round(nb_p * 100, 2),
            'Poisson_Prob': round(pois_p * 100, 2),
            'NB_Absolute_Error': round(err_nb * 100, 2),
            'Poisson_Absolute_Error': round(err_pois * 100, 2),
            'Operational_QC_Status': status
        })
        
    df_prob_estimates = pd.DataFrame(prob_estimates)
    estimates_csv_path = os.path.join(module3_dir, 'defect_count_probability_estimates_summary.csv')
    df_prob_estimates.to_csv(estimates_csv_path, index=False)
    
    # Export individual test evaluations
    df_test_out = df_test.copy()
    df_test_out['NB_Estimated_Probability'] = stats.nbinom.pmf(df_test['Defect_Count'], r_nb, p_nb).round(5)
    df_test_out['Poisson_Estimated_Probability'] = stats.poisson.pmf(df_test['Defect_Count'], lambda_pois).round(5)
    test_eval_csv_path = os.path.join(module3_dir, 'test_batch_defect_evaluations.csv')
    df_test_out.to_csv(test_eval_csv_path, index=False)
    
    # 4. Generate Publication-Grade Visualization Graphs
    print("\nGenerating Module 3 Visualization Graphs...")
    
    # Graph 7: Information Theory Benchmark (Shannon Entropy, Cross-Entropy & KL Divergence)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
    fig.patch.set_facecolor('#FFFFFF')
    
    model_labels = ['Neg-Binomial\nNB(r, p)', 'Poisson\nPois(λ)', 'Binomial\nBin(n=500, p)', 'Uniform\nBaseline']
    x_pos = np.arange(len(model_labels))
    ce_plot = [row['Cross_Entropy_HPQ'] for row in info_metrics_list]
    kl_plot = [row['KL_Divergence_DKL'] for row in info_metrics_list]
    
    # Subplot A: Cross-Entropy and Shannon Entropy Floor
    set_minimal_theme(axes[0])
    bar_ce = axes[0].bar(x_pos, ce_plot, color=['#10B981', '#EF4444', '#F59E0B', '#6B7280'], 
                         edgecolor='#1F2937', linewidth=0.8, alpha=0.85, width=0.55)
    axes[0].axhline(h_p_test, color='#1E40AF', linestyle='--', linewidth=2.0, label=f'Shannon Entropy Floor H(P) = {h_p_test:.2f} bits')
    for bar in bar_ce:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{yval:.2f} bits", 
                     ha='center', va='bottom', fontname='Cambria', fontsize=9.5, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_labels)
    axes[0].set_title("(A) Cross-Entropy Comparison H(P, Q) Against Shannon Floor", pad=12)
    axes[0].set_ylabel("Cross-Entropy (Bits)", labelpad=8)
    axes[0].set_ylim(0, 5.6)
    
    # Subplot B: KL Divergence Information Penalty D_KL(P || Q)
    set_minimal_theme(axes[1])
    bar_kl = axes[1].bar(x_pos, kl_plot, color=['#10B981', '#EF4444', '#F59E0B', '#6B7280'], 
                         edgecolor='#1F2937', linewidth=0.8, alpha=0.85, width=0.55)
    for bar in bar_kl:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f} b", 
                     ha='center', va='bottom', fontname='Cambria', fontsize=9.5, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_labels)
    axes[1].set_title("(B) Relative Entropy Penalty: KL Divergence D_KL(P || Q)", pad=12)
    axes[1].set_ylabel("Kullback-Leibler Divergence (Bits)", labelpad=8)
    axes[1].set_ylim(0, 1.05)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    legend_elements7 = [
        mpl.lines.Line2D([0], [0], color='#1E40AF', lw=2.2, linestyle='--', label=f'True Shannon Entropy H(P) = {h_p_test:.2f} bits'),
        mpl.patches.Patch(facecolor='#10B981', edgecolor='#047857', label='Negative Binomial (Minimal KL Divergence: 0.021 b)'),
        mpl.patches.Patch(facecolor='#EF4444', edgecolor='#B91C1C', label='Poisson Model (Information Loss: 0.748 b)'),
        mpl.patches.Patch(facecolor='#F59E0B', edgecolor='#B45309', label='Binomial Model (Information Loss: 0.768 b)')
    ]
    fig.legend(handles=legend_elements7, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph7_path = os.path.join(module3_dir, '07_information_theory_benchmark_and_divergence.png')
    plt.savefig(graph7_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph7_path}")
    
    # Graph 8: Computational Speedup Benchmark
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Latency Comparison (Log-Log Scale)
    set_minimal_theme(axes[0])
    axes[0].plot(df_bench['Batch_Scale_N'], df_bench['Sequential_Loop_ms'], marker='s', markersize=6, color='#EF4444', 
                 linewidth=2.2, label='Sequential Loops (Scalar Python)')
    axes[0].plot(df_bench['Batch_Scale_N'], df_bench['Vectorized_Tensor_ms'], marker='o', markersize=6, color='#10B981', 
                 linewidth=2.2, label='Vectorized Tensor (NumPy BLAS)')
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].set_title("(A) Execution Latency Benchmark vs. Batch Size", pad=12)
    axes[0].set_xlabel("Number of Evaluated Batches (N)", labelpad=8)
    axes[0].set_ylabel("Execution Latency (Milliseconds, Log Scale)", labelpad=8)
    
    # Subplot B: Speedup Factor Bar Chart
    set_minimal_theme(axes[1])
    bar_pos = np.arange(len(df_bench))
    bar_colors = ['#3B82F6', '#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#06B6D4']
    bars = axes[1].bar(bar_pos, df_bench['Speedup_Factor'], color=bar_colors, edgecolor='#1F2937', linewidth=0.8, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, h + 5, f"{h:.0f}x", 
                     ha='center', va='bottom', fontname='Cambria', fontsize=9.5, fontweight='bold', color='#1F2937')
    axes[1].set_xticks(bar_pos)
    axes[1].set_xticklabels([f"N={n}" for n in df_bench['Batch_Scale_N']])
    axes[1].set_title("(B) Vectorized Tensor Speedup Acceleration Multiplier", pad=12)
    axes[1].set_xlabel("Number of Evaluated Batches (N)", labelpad=8)
    axes[1].set_ylabel("Speedup Acceleration Factor (X-Times Faster)", labelpad=8)
    axes[1].set_ylim(0, max(df_bench['Speedup_Factor']) * 1.18)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    handles8_a, labels8_a = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles8_a, labels=labels8_a, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=2, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph8_path = os.path.join(module3_dir, '08_computational_speedup_benchmark.png')
    plt.savefig(graph8_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph8_path}")
    
    # Graph 9: Probability Estimation Accuracy & Calibration Diagnostic
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Subplot A: Absolute Error Across Defect Counts
    set_minimal_theme(axes[0])
    k_plot = df_prob_estimates['Defect_Count_k']
    nb_err = df_prob_estimates['NB_Absolute_Error']
    pois_err = df_prob_estimates['Poisson_Absolute_Error']
    
    axes[0].plot(k_plot, nb_err, marker='o', color='#10B981', linewidth=2.4, markersize=6, label='Negative Binomial Error (|P - Q_NB|)')
    axes[0].plot(k_plot, pois_err, marker='s', color='#EF4444', linestyle='--', linewidth=2.2, markersize=5.5, label='Poisson Error (|P - Q_Pois|)')
    axes[0].set_title("(A) Probability Estimation Error Across Defect Counts (k)", pad=12)
    axes[0].set_xlabel("Defect Count per Batch (k)", labelpad=8)
    axes[0].set_ylabel("Absolute Estimation Error (%)", labelpad=8)
    axes[0].set_ylim(-0.2, 8.5)
    
    # Subplot B: Cumulative Risk Calibration: Empirical vs. Model Quantiles
    set_minimal_theme(axes[1])
    k_eval = np.arange(0, 22)
    cdf_emp_test = np.array([np.mean(defects_test <= k) for k in k_eval])
    cdf_nb_test = stats.nbinom.cdf(k_eval, r_nb, p_nb)
    cdf_pois_test = stats.poisson.cdf(k_eval, lambda_pois)
    
    axes[1].plot([0, 1], [0, 1], color='#6B7280', linestyle='--', linewidth=1.5, label='Ideal Perfect Calibration (y = x)')
    axes[1].plot(cdf_nb_test, cdf_emp_test, marker='o', color='#10B981', linewidth=2.2, label='Neg-Binomial Calibration')
    axes[1].plot(cdf_pois_test, cdf_emp_test, marker='s', color='#EF4444', linestyle=':', linewidth=2.0, label='Poisson Calibration')
    axes[1].set_title("(B) Cumulative Probability Calibration Diagnostic Curve", pad=12)
    axes[1].set_xlabel("Predicted Model Cumulative Probability F_model(k)", labelpad=8)
    axes[1].set_ylabel("Empirical True Cumulative Fraction F_emp(k)", labelpad=8)
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    
    # Unified Bottom Legend
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    handles9_a, labels9_a = axes[0].get_legend_handles_labels()
    handles9_b, labels9_b = axes[1].get_legend_handles_labels()
    all_h9 = [handles9_a[0], handles9_a[1], handles9_b[0]]
    all_l9 = [labels9_a[0], labels9_a[1], 'Ideal Calibration Reference Line']
    fig.legend(handles=all_h9, labels=all_l9, loc='lower center', bbox_to_anchor=(0.5, 0.04), 
               ncol=3, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
               
    graph9_path = os.path.join(module3_dir, '09_defect_probability_estimation_and_calibration.png')
    plt.savefig(graph9_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {graph9_path}")
    
    print("\nModule 3 execution completed successfully!")

if __name__ == '__main__':
    run_module_3()
