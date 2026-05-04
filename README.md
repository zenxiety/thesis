# Brain Tumor Classification Model for MRI Images Using Local Binary Pattern Variants and Support Vector Machine with Lime-Based Interpretability Analysis

[Deployed Streamlit Model Demo](https://andyanyoga-thesis.streamlit.app)

## Pipeline Overview

1. Data Acquisition
2. Dataset Merge (Train + Test)
3. Train-Test Split
4. Data Preprocessing
5. Feature Extraction
6. Modelling & Classification
7. Model Evaluation
8. Explainable AI

## Dataset

- Source: Kaggle - [Brain Tumor MRI Dataset by SARTAJ](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- Classes:
    - `glioma_tumor`
    - `meningioma_tumor`
    - `no_tumor`
    - `pituitary_tumor`

## Feature Extractions

| Code              | Method                         | Description                                                                      |
| ----------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| `lbplib`          | Library LBP (scikit-image)     | LBP implementation using `nri_uniform` mode                                      |
| `lbp`             | Classic LBP                    | Basic LBP implementation without uniform pattern filtering                       |
| `lbpu`            | Uniform LBP                    | Classic LBP from scratch with uniform pattern filtering                          |
| `nlbp1–4`         | Neighbor LBP                   | Pairwise comparison among neighboring pixels with varying distances (1-4 pixels) |
| `albp0/45/90/135` | Angular LBP                    | Directional LBP based on angular relationships (0°, 45°, 90°, 135°)              |
| `mblbp2–4`        | Multi-Block LBP                | Region-based LBP using average intensity via integral image                      |
| `semblbp2–4`      | Statistically Effective MB-LBP | MB-LBP extension with histogram simplification based on frequency patterns       |

## Classifier Models

### Machine Learning

| Model                    | Hyperparamters                                |
| ------------------------ | --------------------------------------------- |
| K-Nearest Neighbors      | `n_neighbors`, `weights`, `metric`            |
| Decision Tree            | `criterion`, `max_depth`, `min_samples_split` |
| Random Forest            | `n_estimators`, `criterion`, `max_depth`      |
| Multinomial Naive Bayes  | `alpha`, `fit_prior`                          |
| Gaussian Naive Bayes     | `var_smoothing`                               |
| SVM — Linear (LinearSVC) | `C`                                           |
| SVM — Linear (SVC)       | `C`, `decision_function_shape`                |
| SVM — Polynomial         | `C`, `degree`, `gamma`, `coef0`               |
| SVM — Sigmoid            | `C`, `gamma`, `coef0`                         |
| SVM — RBF                | `C`, `gamma`                                  |

### Deep Learning

CNN Architecture:

- Convolution Blocks

    4 Conv2D + MaxPooling (32 → 64 → 128 → 256 filters)

- Fully Connected Layers

    Dense: 512 → 256 → 128 → 4 (Softmax)

- Regularizations

    Dropout: 0.5 and 0.2

- Training Configurations
    - Optimizer Parameters: Adam
    - Learning Rate: 0.001
    - Loss: Sparse Categorical Crossentropy

## Explainable AI — LIME

- Method: `LimeImageExplainer()` from `lime_image`
- Model: Best SVM Kernel + Feature Extraction Method with Probability
- Output:
    - Superpixel-based explanation
    - Visual attribution highlighting regions influencing classification decisions

## Result

| Model            | Method   | Macro Accuracy | F1-Score  |
| ---------------- | -------- | -------------- | --------- |
| KNN              | nLBP1    | 87.29%         | 0.872     |
| DT               | nLBP1    | 0.740          | 0.767     |
| RF               | nLBP2    | 0.835          | 0.844     |
| MNB              | nLBP1    | 53.14%         | 0.509     |
| GNB              | nLBP2    | 60.79%         | 0.587     |
| SVM LinearSVC    | nLBP1    | 75.04%         | 0.755     |
| SVM Linear (SVC) | nLBP1    | 76.72%         | 0.771     |
| SVM Polynomial   | nLBP1    | 86.68%         | 0.869     |
| SVM Sigmoid      | nLBP4    | 66.92%         | 0.674     |
| **SVM RBF**      | **LBPu** | **88.67%**     | **0.886** |
| CNN              | -        | 86.52%         | 0.861     |
