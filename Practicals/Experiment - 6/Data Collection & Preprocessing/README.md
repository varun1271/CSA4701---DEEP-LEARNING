# Module 1: Data Collection & Preprocessing

## 1. Module Overview
Module 1 generates an ill-conditioned neural network optimization benchmark designed to induce pathological loss curvature and severe gradient oscillations.

In this scenario:
- **Total Observations:** $N = 1,000$ signal records
- **Predictor Dimensions:** $D = 8$ engineered continuous physical features exhibiting extreme variance disparity and multicollinearity ($r > 0.95$)
- **Hessian Curvature Anisotropy:** Maximum eigenvalue $\lambda_{\max} = 1,692.57$ versus minimum eigenvalue $\lambda_{\min} = 0.0279$, yielding an extreme condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \approx 60,772.7 \gg 1$
- **Primary Research Focus:** Demonstrating how pathological ravines cause standard gradient descent to oscillate uncontrollably across the steep canyon walls, and proving how **Momentum Gradient Descent** eliminates these cross-axis oscillations while compounding longitudinal velocity.

---

## 2. Feature Architecture & Curvature Profile

| Feature Index | Feature Name | Measurement Unit | Domain / Sensor Type | Optimization Curvature Role |
| :---: | :--- | :---: | :--- | :--- |
| **F01** | `Signal_Amplitude_HighVar` | Volts ($\text{V}$) | High-Power Sensor | High variance ($\sigma = 41.8\text{ V}$); aligns with the steepest canyon axis. |
| **F02** | `Frequency_Modulation_LowVar` | Radians / sec | Carrier Demodulator | Extremely low variance ($\sigma = 0.35$); aligns with the shallow ravine floor. |
| **F03** | `Phase_Shift_Collinear1` | Radians | Primary Phase Detector | Harmonic offset ($0.1$ to $6.2\text{ rad}$). Collinear with Phase2 ($r = 0.96$). |
| **F04** | `Phase_Shift_Collinear2` | Radians | Secondary Phase Detector | Induces sharp diagonal ill-conditioning in parameter space. |
| **F05** | `Harmonic_Resonance_Ratio` | Dimensionless | Spectral Analyzer | Non-linear harmonic power ratio ($0.1$ to $4.0$). |
| **F06** | `Damping_Factor_Proxy` | $\text{sec}^{-1}$ | Dissipation Metrology | Exponential dissipation rate ($0.2$ to $5.5\text{ s}^{-1}$). |
| **F07** | `Noise_Floor_Jitter` | Dimensionless | Channel Interference | Additive stochastic perturbations ($\sigma = 0.5$). |
| **F08** | `Spectral_Centroid_Offset` | Hertz ($\text{Hz}$) | Audio / RF Centroid | Spectral energy distribution center of mass. |

---

## 3. Mathematical Foundations: The Ill-Conditioned Ravine

When expanding the loss $\mathcal{L}(\mathbf{w})$ via Taylor series about a local minimum $\mathbf{w}^*$:
$$\mathcal{L}(\mathbf{w}) \approx \mathcal{L}(\mathbf{w}^*) + \frac{1}{2} (\mathbf{w} - \mathbf{w}^*)^T \mathbf{H} (\mathbf{w} - \mathbf{w}^*)$$
where $\mathbf{H} = \nabla^2 \mathcal{L}(\mathbf{w}^*)$ is the Hessian matrix.
- When condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \gg 1$:
  - Along the eigenvector $\mathbf{u}_{\max}$ (steep canyon wall), gradient magnitude is proportional to $\lambda_{\max}$. A step size $\eta > \frac{2}{\lambda_{\max}}$ triggers exponential divergence (instability).
  - Along the eigenvector $\mathbf{u}_{\min}$ (longitudinal canyon floor), progress per iteration is bounded by $\eta \lambda_{\min} \le \frac{2 \lambda_{\min}}{\lambda_{\max}} = \frac{2}{\kappa} \ll 1$.
  - Consequently, standard gradient descent is forced into severe perpendicular zig-zagging, making virtually zero forward progress.

---

## 4. Artifacts and Generated Data Files

- **Raw Dataset:** [raw_ill_conditioned_signals_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Collection%20&%20Preprocessing/raw_ill_conditioned_signals_1000.csv)
- **Preprocessed Dataset:** [preprocessed_optimization_signals_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Collection%20&%20Preprocessing/preprocessed_optimization_signals_1000.csv)
- **Feature Metadata & Summary:** [signal_feature_metadata_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Collection%20&%20Preprocessing/signal_feature_metadata_1000.csv)
- **Visualizations:**
  - [01_ill_conditioned_feature_distributions.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Collection%20&%20Preprocessing/01_ill_conditioned_feature_distributions.png)
  - [02_feature_correlation_and_eigenvalue_spectrum.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Collection%20&%20Preprocessing/02_feature_correlation_and_eigenvalue_spectrum.png)
  - [03_feature_standardization_and_outlier_bounds.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Collection%20&%20Preprocessing/03_feature_standardization_and_outlier_bounds.png)
