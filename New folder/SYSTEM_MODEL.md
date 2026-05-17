# Complete System Model & Mathematical Foundations

**Project:** Heart Disease Prediction Using Machine Learning Algorithms — A Comparative Study with Explainable AI

---

## 1. Complete System Architecture

### 1.1 High-Level Block Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                  │
│  UCI Cleveland Heart Disease Dataset (303 patients × 14 cols)    │
│  13 clinical features + 1 target (0=no disease, 1-4=disease)     │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   DATA PREPROCESSING LAYER                       │
│  • Load CSV with proper column names                             │
│  • Convert target to binary (0 = healthy, 1 = disease)           │
│  • Impute missing values (mode for ca, thal columns)             │
│  • Save cleaned data to data/processed/                          │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                       │
│  • Numerical features: age, trestbps, chol, thalach, oldpeak     │
│    → Standardize: z = (x - μ) / σ                                │
│  • Categorical features: sex, cp, fbs, restecg, exang,           │
│    slope, ca, thal                                               │
│    → One-Hot Encoding                                            │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  DATA SPLITTING LAYER                            │
│  • Stratified Split: 80% Train | 20% Test                        │
│  • SMOTE applied to TRAIN ONLY (test stays untouched)            │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MODEL LAYER (9 Models)                        │
│                                                                  │
│  CLASSIFIERS (6):           REGRESSORS (3, thresholded):         │
│  • Logistic Regression      • Decision Tree Regressor            │
│  • Decision Tree Classifier • Random Forest Regressor            │
│  • Random Forest Classifier • Support Vector Regressor           │
│  • Support Vector Classifier                                     │
│  • K-Nearest Neighbors                                           │
│  • Gaussian Naive Bayes                                          │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              HYPERPARAMETER TUNING LAYER                         │
│  • GridSearchCV                                                  │
│  • 10-Fold Stratified Cross-Validation                           │
│  • Scoring metric: ROC-AUC                                       │
│  • ~1000+ candidate models fitted                                │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  EVALUATION LAYER                                │
│  • Accuracy, Precision, Recall, F1-Score, ROC-AUC                │
│  • Confusion Matrix, ROC Curves, PR Curves                       │
│  • Bar chart comparison across all 9 models                      │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│            EXPLAINABLE AI (XAI) LAYER                            │
│  • SHAP (SHapley Additive exPlanations)                          │
│  • Global: which features matter overall                         │
│  • Local: why this specific patient got this prediction          │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              DEPLOYMENT LAYER (Streamlit Web App)                │
│  • Live patient input form                                       │
│  • Risk percentage with color-coded gauge                        │
│  • SHAP per-patient explanation                                  │
│  • Model comparison dashboard                                    │
│  • Public link: heart-disease-xai.streamlit.app                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Description

| Feature       | Type        | Description                                   |
|---------------|-------------|-----------------------------------------------|
| `age`         | Numeric     | Patient age in years                          |
| `sex`         | Categorical | 0 = female, 1 = male                          |
| `cp`          | Categorical | Chest pain type (1–4)                         |
| `trestbps`    | Numeric     | Resting blood pressure (mm Hg)                |
| `chol`        | Numeric     | Serum cholesterol (mg/dl)                     |
| `fbs`         | Categorical | Fasting blood sugar > 120 mg/dl (0/1)         |
| `restecg`     | Categorical | Resting ECG (0/1/2)                           |
| `thalach`     | Numeric     | Maximum heart rate achieved                   |
| `exang`       | Categorical | Exercise-induced angina (0/1)                 |
| `oldpeak`     | Numeric     | ST depression induced by exercise             |
| `slope`       | Categorical | Slope of peak exercise ST segment (1/2/3)     |
| `ca`          | Categorical | Major vessels (0–3) colored by fluoroscopy    |
| `thal`        | Categorical | Thalassemia (3/6/7)                           |
| `target`      | Binary      | 0 = No disease, 1 = Disease                   |

---

## 3. Mathematical Foundations of All 9 Models

### Notation
- **x** = feature vector (13-dimensional after preprocessing)
- **y** = true label (0 or 1)
- **ŷ** = predicted label
- **P(y=1|x)** = probability of disease given features
- **w**, **b** = model parameters (weights and bias)

---

### 3.1 Logistic Regression (LR)

**Type:** Linear classifier
**Idea:** Fit a linear boundary, squash output through sigmoid to get probability.

#### Equation
The sigmoid function:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The prediction:
$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

**Loss function** (Binary Cross-Entropy):
$$
L = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]
$$

**Graph:** Sigmoid S-curve. Output between 0 and 1.

```
   1.0 ┤            ─────────
       │         ╱
   0.5 ┤      ╱
       │   ╱
   0.0 ┤─────
       └──────────────────────
              0 (boundary)
```

---

### 3.2 Decision Tree (Classifier & Regressor)

**Type:** Tree-based, non-linear
**Idea:** Recursively split data on feature thresholds to maximize purity.

#### Equation — Splitting criterion

**Gini Impurity** (default):
$$
G = 1 - \sum_{k=1}^{K} p_k^2
$$

**Information Gain (Entropy):**
$$
H = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

Where $p_k$ = proportion of class $k$ at the node.

**For regression** (DT Regressor uses MSE):
$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \bar{y})^2
$$

**Graph:** Tree diagram with conditional branches.

```
         [cp ≤ 3.5?]
         /          \
      Yes           No
      /              \
[ca ≤ 0.5?]      [oldpeak ≤ 1.8?]
   /  \              /  \
  ...  ...         ...  ...
```

---

### 3.3 Random Forest (Classifier & Regressor)

**Type:** Ensemble of Decision Trees (Bagging)
**Idea:** Train many trees on bootstrap samples with random feature subsets, then aggregate.

#### Equation

For classification (majority vote):
$$
\hat{y} = \text{mode}\{T_1(\mathbf{x}), T_2(\mathbf{x}), \dots, T_M(\mathbf{x})\}
$$

For regression (mean):
$$
\hat{y} = \frac{1}{M}\sum_{m=1}^{M} T_m(\mathbf{x})
$$

Where $M$ = number of trees (200 in our project), $T_m$ = the m-th tree.

**Graph:** Multiple trees voting.

```
   Tree 1 →  [Disease]   ┐
   Tree 2 →  [Disease]   │
   Tree 3 →  [Healthy]   ├─→  Majority Vote → [Disease]
   ...                   │
   Tree 200→ [Disease]   ┘
```

---

### 3.4 Support Vector Classifier (SVC)

**Type:** Margin-based classifier
**Idea:** Find the hyperplane that maximizes the margin between classes.

#### Equation

Decision function:
$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \phi(\mathbf{x}) + b)
$$

Optimization problem (with RBF kernel):
$$
\min_{\mathbf{w}, b, \xi} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N}\xi_i
$$

Subject to:
$$
y_i(\mathbf{w}^T \phi(\mathbf{x}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

**RBF Kernel:**
$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)
$$

**Graph:** Hyperplane separating two classes with maximum margin.

```
   Class 1  ●  ●  ●
                  ●─────────── (margin)
   ━━━━━━━━━━━━━━━━━━━━ (hyperplane)
                  ───────────── (margin)
              ○  ○
   Class 2  ○  ○
```

---

### 3.5 Support Vector Regressor (SVR)

**Type:** Regression version of SVM
**Idea:** Find a function within ε-tube around training points.

#### Equation

$$
f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b
$$

Optimization:
$$
\min_{\mathbf{w}, b, \xi, \xi^*} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N}(\xi_i + \xi_i^*)
$$

Subject to:
$$
y_i - \mathbf{w}^T \phi(\mathbf{x}_i) - b \leq \varepsilon + \xi_i
$$
$$
\mathbf{w}^T \phi(\mathbf{x}_i) + b - y_i \leq \varepsilon + \xi_i^*
$$

**Graph:** ε-tube around the regression line.

```
    y
    │       ╱
    │    ╱  · · · · · ε-tube (upper)
    │ ╱ · · · · · · · · prediction line
    │· · · · · · · · ε-tube (lower)
    │
    └────────────── x
```

---

### 3.6 K-Nearest Neighbors (KNN)

**Type:** Instance-based, lazy learner
**Idea:** Predict class by majority vote of k closest training points.

#### Equation

**Euclidean distance:**
$$
d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{f=1}^{F}(x_{i,f} - x_{j,f})^2}
$$

**Manhattan distance** (alternative):
$$
d(\mathbf{x}_i, \mathbf{x}_j) = \sum_{f=1}^{F}|x_{i,f} - x_{j,f}|
$$

**Prediction:**
$$
\hat{y} = \text{mode}\{y_{n_1}, y_{n_2}, \dots, y_{n_k}\}
$$
Where $n_1, \dots, n_k$ are the indices of the $k$ closest training samples.

**Graph:** Query point and its k nearest neighbors.

```
   ●    ●          ●
        ●    ★      (★ = new patient, k=5)
         ○ ○        Neighbors: 3 ● + 2 ○ → predicted: ●
         ○
   ○
```

---

### 3.7 Gaussian Naive Bayes (NB)

**Type:** Probabilistic classifier
**Idea:** Apply Bayes' Theorem assuming feature independence.

#### Equation

**Bayes' Theorem:**
$$
P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})}
$$

**Naive assumption** (features are independent):
$$
P(\mathbf{x} \mid y) = \prod_{f=1}^{F} P(x_f \mid y)
$$

**Gaussian likelihood for each feature:**
$$
P(x_f \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}}\exp\left(-\frac{(x_f - \mu_y)^2}{2\sigma_y^2}\right)
$$

**Final prediction:**
$$
\hat{y} = \arg\max_{y \in \{0, 1\}} P(y)\prod_{f=1}^{F} P(x_f \mid y)
$$

**Graph:** Two Gaussian curves, one per class.

```
   P(x|y)
    │      ╱╲ Healthy (y=0)
    │     ╱  ╲
    │    ╱ ╱╲ ╲ Disease (y=1)
    │   ╱ ╱  ╲ ╲
    │  ╱ ╱    ╲ ╲
    └─────────────── feature x
```

---

## 4. SHAP (Explainable AI) Mathematics

### 4.1 Shapley Value (from Game Theory)

Each feature contributes some "value" to the prediction. SHAP fairly distributes the prediction among features.

$$
\phi_f = \sum_{S \subseteq F \setminus \{f\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!}\left[v(S \cup \{f\}) - v(S)\right]
$$

Where:
- $\phi_f$ = SHAP value of feature $f$
- $F$ = set of all features
- $S$ = subset of features not containing $f$
- $v(S)$ = model prediction using only features in $S$

### 4.2 Additive Property

$$
\hat{y} = \phi_0 + \sum_{f=1}^{F} \phi_f
$$

Where $\phi_0$ = baseline (average prediction across all patients).

**Interpretation:** Each $\phi_f$ tells you how much feature $f$ pushed THIS prediction up or down from the baseline.

---

## 5. SMOTE (Class Balancing) Mathematics

To balance classes, SMOTE creates synthetic minority samples:

For each minority sample $\mathbf{x}_i$:
1. Find $k$ nearest minority-class neighbors
2. Pick one randomly, call it $\mathbf{x}_j$
3. Generate synthetic sample:
$$
\mathbf{x}_{new} = \mathbf{x}_i + \lambda(\mathbf{x}_j - \mathbf{x}_i), \quad \lambda \in [0, 1]
$$

**Graph:**

```
       Minority points: ●
              ●         ●
                  ⊕  ← synthetic (created on line between two ●)
              ●         ●
```

---

## 6. Evaluation Metrics (Equations)

### 6.1 Confusion Matrix

```
                      Predicted
                   No        Yes
              ┌─────────┬─────────┐
   Actual No  │   TN    │   FP    │
              ├─────────┼─────────┤
   Actual Yes │   FN    │   TP    │
              └─────────┴─────────┘
```

### 6.2 Metrics

**Accuracy** (overall correctness):
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Precision** (of those predicted positive, how many were correct):
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Recall / Sensitivity** (of actual positives, how many we caught):
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1-Score** (harmonic mean of Precision and Recall):
$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**ROC-AUC:** Area under the ROC curve (TPR vs FPR across thresholds). 1.0 = perfect, 0.5 = random.

---

## 7. Cross-Validation Mathematics

**10-Fold Stratified CV:**

$$
\text{CV-Score} = \frac{1}{10}\sum_{k=1}^{10}\text{Score}(M_k, D_{test}^{(k)})
$$

Where $M_k$ = model trained on 9 folds, $D_{test}^{(k)}$ = the held-out 10th fold.

**Graph:**
```
Fold 1:  [TEST][ TRAIN................................]
Fold 2:  [    ][TEST][ TRAIN............................]
Fold 3:  [    ][    ][TEST][ TRAIN........................]
...
Fold 10: [ TRAIN................................][TEST]

Average score across 10 folds → CV-Score
```

---

## 8. Complete Final Results Table

| Rank | Model                          | Accuracy | Precision | Recall | F1     | ROC-AUC |
|------|--------------------------------|---------:|----------:|-------:|-------:|--------:|
| 🥇 1 | Logistic Regression            |   0.869  |   0.813   | 0.964  | 0.867  | **0.966** |
| 🥇 1 | Support Vector Classifier      | **0.902**|   0.844   | 0.964  | **0.900** | **0.966** |
| 3   | KNN                            | **0.902**| **0.867** | 0.929  | 0.897  | 0.948   |
| 4   | Support Vector Regressor       |   0.852  |   0.806   | 0.893  | 0.847  | 0.948   |
| 5   | Random Forest (Classifier)     |   0.852  |   0.806   | 0.893  | 0.847  | 0.946   |
| 6   | Random Forest (Regressor)      |   0.885  |   0.818   | 0.964  | 0.885  | 0.929   |
| 7   | Decision Tree (Classifier)     |   0.836  |   0.800   | 0.857  | 0.828  | 0.853   |
| 7   | Decision Tree (Regressor)      |   0.836  |   0.800   | 0.857  | 0.828  | 0.853   |
| 9   | Gaussian Naive Bayes           |   0.656  |   0.581   | 0.893  | 0.704  | 0.827   |

---

## 9. Generated Graphs (already saved in `results/figures/`)

| # | File                                   | What it shows                             |
|---|----------------------------------------|-------------------------------------------|
| 1 | `01_class_distribution.png`            | How many patients have disease vs healthy |
| 2 | `02_correlation_heatmap.png`           | Correlation between all 13 features       |
| 3 | `03_feature_distributions.png`         | Histograms of numeric features by class   |
| 4 | `04_confusion_matrices.png`            | All 9 model confusion matrices            |
| 5 | `05_roc_curves.png`                    | ROC curves overlaid for all 9 models      |
| 6 | `06_pr_curves.png`                     | Precision-Recall curves for all 9         |
| 7 | `07_metrics_comparison.png`            | Bar chart: 9 models × 5 metrics           |
| 8 | `08_best_model_summary.png`            | Winner's metric breakdown                 |
| 9 | `09_shap_bar_logistic_regression.png`  | Global SHAP: which features matter most   |
| 10| `10_shap_summary_logistic_regression.png` | SHAP beeswarm: feature impact distribution |

---

## 10. How Everything Connects (Workflow)

```
1. RAW DATA  →  cleveland.csv (303 patients, 14 columns)
        │
        ▼
2. CLEAN     →  Binary target, mode imputation
        │
        ▼
3. PROCESS   →  z-score scaling + one-hot encoding
        │
        ▼
4. SPLIT     →  80% train, 20% test (stratified)
        │
        ▼
5. BALANCE   →  SMOTE on training data only
        │
        ▼
6. TUNE      →  For each of 9 models, GridSearchCV with 10-fold CV
        │           - Search ~1000 combinations
        │           - Optimize ROC-AUC
        ▼
7. EVALUATE  →  Test set: Accuracy, Precision, Recall, F1, AUC
        │
        ▼
8. EXPLAIN   →  SHAP for global + per-patient
        │
        ▼
9. DEPLOY    →  Streamlit web app (live demo)
```

---

## 11. Summary of Key Concepts You'll Be Asked

| Question | Short Answer |
|----------|--------------|
| What is supervised learning? | Learning from labeled examples (input → output) |
| What is classification vs regression? | Classification = discrete labels; Regression = continuous values |
| Why cross-validation? | Reduces variance, more reliable performance estimate |
| Why SMOTE? | Balances minority class without simple duplication |
| Why SHAP? | Explains black-box model predictions feature-by-feature |
| Why so many models? | To compare and find the best for this specific problem |
| Why Logistic Regression won? | Small dataset → simpler models generalize better |
| Why Naive Bayes lost? | Independence assumption violated (cardiac features are correlated) |

---

**Author:** Aqib · BS Robotics and Intelligent Systems
**Course:** CSC 412 — Artificial Intelligence (Spring 2026)
**Institution:** Bahria University, H-11 Campus, Islamabad
