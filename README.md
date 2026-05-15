# Heart Disease Risk Prediction with Explainable AI

**Course:** CSC 412 – Artificial Intelligence (Spring 2026)
**Domain:** Healthcare & Bioinformatics (SDG 3 – Good Health & Well-being)
**Institution:** Bahria University, H-11 Campus, Islamabad

An end-to-end explainable AI framework that compares **7 machine learning algorithms** for cardiovascular disease risk prediction, with **SHAP** for transparent explanations, all wrapped in a **Streamlit web app**.

---

## Highlights

- **7 algorithms:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, MLP Neural Network
- **GridSearchCV** with **10-fold stratified cross-validation**
- **SMOTE** for class balance
- **SHAP** for global + per-patient explanations
- **Streamlit web app** for live predictions
- **IEEE-format research paper** ready for submission

## Results

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|---------:|----------:|-------:|-------:|--------:|
| Logistic Regression |   0.869  |   0.813   | 0.964  | 0.867  | **0.966** |
| SVM (RBF)           | **0.902**|   0.844   | 0.964  | **0.900** | **0.966** |
| Neural Network (MLP)|   0.869  |   0.833   | 0.893  | 0.862  | 0.950   |
| KNN                 | **0.902**| **0.867** | 0.929  | 0.897  | 0.948   |
| Random Forest       |   0.852  |   0.806   | 0.893  | 0.847  | 0.946   |
| XGBoost             |   0.885  |   0.862   | 0.893  | 0.877  | 0.936   |
| Decision Tree       |   0.836  |   0.800   | 0.857  | 0.828  | 0.853   |

## Project structure

```
heart_disease_xai/
├── data/
│   ├── raw/cleveland.csv              UCI dataset
│   └── processed/cleveland_clean.csv  cleaned + binary target
├── src/
│   ├── config.py                      central settings
│   ├── data_loader.py                 load + clean
│   ├── preprocessing.py               scale + encode + SMOTE
│   ├── models.py                      7 algorithms + grids
│   ├── tuning.py                      GridSearchCV + 10-fold CV
│   ├── evaluation.py                  metrics + comparison
│   ├── visualize.py                   8 plots
│   ├── explainability.py              SHAP global + local
│   └── train.py                       end-to-end orchestrator
├── app/
│   └── streamlit_app.py               interactive web demo
├── results/
│   ├── figures/                       all generated plots
│   ├── metrics.csv                    final scorecard
│   └── trained_models/                joblib artifacts
├── report/
│   └── report.tex                     IEEE-format paper
├── requirements.txt
└── README.md
```

## How to run

### 1. Train every model end-to-end
```powershell
cd "heart_disease_xai"
python -m src.train
```
Tunes all 7 models with 10-fold CV, evaluates them on the held-out 20% test set, and writes plots + metrics under `results/`. Takes ~4 minutes on a laptop.

### 2. Launch the web app
```powershell
streamlit run app/streamlit_app.py
```
Then open http://localhost:8501 in your browser.

### 3. Build the paper
Open `report/report.tex` on Overleaf or compile locally with `pdflatex`. Figures referenced from the paper live under `results/figures/`.

## Methodology block diagram

```
[UCI Cleveland Data] -> [Cleaning + Imputation] -> [Scaling + One-Hot]
   -> [80/20 Stratified Split] -> [SMOTE on train] -> [GridSearchCV x 7 models]
   -> [10-fold CV] -> [Evaluation] -> [SHAP Explanations] -> [Streamlit Web App]
```

## Dataset

- **UCI Cleveland Heart Disease** (303 patients, 13 features)
- Target collapsed from 5-class severity (0–4) into binary (no-disease / disease)
- Available at: <https://archive.ics.uci.edu/ml/datasets/heart+disease>

## License

Educational use only.

## Author

Aqib · BS Robotics and Intelligent Systems · maqib9254@gmail.com
