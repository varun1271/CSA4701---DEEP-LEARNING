# CSA4701 - Deep Learning

**Student Name:** Y. Varun Kumar Reddy  
**Registration Number:** 192424230  
**Course:** Deep Learning (CSA4701)  

---

## Repository Structure

```
CSA4701---DEEP-LEARNING/
└── Practicals/
    └── Experiment - 1/
        ├── Experiment 1 Report.pdf
        ├── Data Collection & Preprocessing/
        │   ├── 01_data_collection_preprocessing.py
        │   ├── raw_products_500.csv
        │   ├── preprocessed_products_500.csv
        │   ├── products_metadata_500.csv
        │   ├── 01_before_after_feature_scales.png
        │   ├── 02_before_after_distribution_kde.png
        │   ├── 03_before_after_correlation_heatmap.png
        │   └── README.md
        ├── Data Splitting & Model Dessign/
        │   ├── 02_data_splitting_model_design.py
        │   ├── train_interactions.csv
        │   ├── val_interactions.csv
        │   ├── test_interactions.csv
        │   ├── customer_preference_matrix.npy
        │   ├── product_catalog_matrix.npy
        │   ├── 04_data_splitting_proportions.png
        │   ├── 05_product_similarity_matrix_heatmap.png
        │   ├── 06_vector_dot_vs_matrix_mult_architecture.png
        │   └── README.md
        └── Training & Evaluation/
            ├── 03_training_and_evaluation.py
            ├── benchmark_timing_results.csv
            ├── evaluation_metrics_summary.csv
            ├── top_k_recommendations_sample.csv
            ├── 07_evaluation_metrics_across_k.png
            ├── 08_computational_speedup_benchmark.png
            ├── 09_recommendation_score_distribution.png
            └── README.md
```

---

## Experiment 1: E-Commerce Recommendation System
**Scenario:** An online shopping platform represents 500 products using 20 numerical features each. Vector dot products and matrix multiplication are utilized to calculate similarity scores and recommend relevant products to customers.

### Modules:
1. **Module 1: Data Collection & Preprocessing**
   - Feature engineering, outlier handling, Standard Scaling, and L2 Unit-Norm Normalization to ensure vector dot products compute Cosine Similarity $\in [-1, 1]$.
2. **Module 2: Data Splitting & Model Design**
   - Customer preference matrix synthesis ($\mathbf{U} \in \mathbb{R}^{100 \times 20}$), product catalog matrix ($\mathbf{P} \in \mathbb{R}^{500 \times 20}$), 70/15/15 interaction splitting, and similarity formulation.
3. **Module 3: Training & Evaluation**
   - Computational speedup benchmarking (sequential dot product loops vs vectorized matrix multiplication $\mathbf{U}\mathbf{P}^T$) and Top-K recommendation ranking evaluation (Precision, Recall, NDCG, Hit Rate, MRR, Coverage).
