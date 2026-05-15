# 10-Minute Presentation Script

## Slide 1 (0:00 - 0:45) — Title & Hook
**Say:**
> "Good morning everyone. My project is titled *An Explainable AI Framework for Cardiovascular Disease Risk Prediction*. According to the WHO, heart disease kills almost 18 million people every year — making it the world's number one killer. In Pakistan alone, it's responsible for nearly 30% of all deaths. My goal was to build an AI system that doesn't just predict heart disease, but also *explains* why — because doctors need to trust the model before they use it."

---

## Slide 2 (0:45 - 1:45) — Problem & Motivation
**Say:**
> "There are two big problems with existing AI medical models. First, most papers compare only one or two algorithms, so we don't know which is really best. Second — and more critical — modern models are black boxes. A doctor can't trust a number on a screen unless they understand *why* the AI made that decision. My project solves both issues."

**Show:** problem statement slide with two bullet points.

---

## Slide 3 (1:45 - 3:00) — Block Diagram
**Say:**
> "Here's how my system works end-to-end. I start with the UCI Cleveland Heart Disease dataset — 303 real patients with 13 clinical features. The data goes through cleaning, scaling, and one-hot encoding. I split 80/20, then apply SMOTE to balance the classes. Then I run hyperparameter tuning with 10-fold cross-validation across seven different ML algorithms. After evaluation, I use SHAP — Shapley Additive Explanations — to make the predictions transparent. Finally, everything is deployed as a Streamlit web app."

**Show:** the block diagram from `results/figures/` or the one in your report.

---

## Slide 4 (3:00 - 4:30) — Models & Tuning
**Say:**
> "I compared seven models, ranging from simple Logistic Regression to advanced XGBoost and a Neural Network. For each one, I used GridSearchCV with 10-fold stratified cross-validation, optimizing for ROC-AUC. In total over 1,500 candidate models were fitted. Class imbalance was handled with SMOTE — Synthetic Minority Oversampling — applied only on the training set, so the test set stays unbiased."

**Show:** list of 7 models + metrics table.

---

## Slide 5 (4:30 - 5:30) — Results
**Say:**
> "Here are the test-set results. Logistic Regression and SVM both achieved an AUC of 0.966 and 90% accuracy — that's better than what most published papers on this dataset report. Recall is 96.4%, meaning the system correctly identifies almost every at-risk patient. That's the most important metric in a medical screening context."

**Show:** `07_metrics_comparison.png` and `05_roc_curves.png`.

---

## Slide 6 (5:30 - 6:30) — Explainability with SHAP
**Say:**
> "Now, the most important part — explainability. SHAP tells me which features matter most. The top predictors my model learned are exactly what cardiologists would expect: chest pain type, number of blocked vessels, maximum heart rate, ST depression, and thalassemia status. This proves my model isn't relying on spurious correlations — it has actually learned valid medical patterns."

**Show:** `09_shap_bar_logistic_regression.png`.

---

## Slide 7 (6:30 - 8:30) — LIVE DEMO
**Say (while running):**
> "Let me show you the live web app."

```powershell
streamlit run app/streamlit_app.py
```

**Demo flow:**
1. Open the app — show the sidebar with patient inputs.
2. Set a "high-risk patient": Age 65, Male, asymptomatic chest pain (type 4), 3 major vessels, oldpeak 3.5.
3. Click around — show the risk gauge turning red, the percentage.
4. Scroll down — show the SHAP bar chart explaining the prediction.
5. Click the "Model Comparison" tab — show all the plots.
6. Click "Data Insights" — show the correlation heatmap.

**Say:**
> "Notice how the app doesn't just say 'high risk' — it tells the doctor exactly *which* features contributed and by how much. That's clinical-grade transparency."

---

## Slide 8 (8:30 - 9:30) — Research Paper & Contributions
**Say:**
> "I wrote a 4-page IEEE-format research paper documenting everything: methodology, results, related work, and SHAP analysis. My main contributions are: rigorous 7-algorithm comparison under identical conditions, integrating SHAP explainability into the prediction pipeline, and deploying the whole thing as an interactive web app. This is publishable at undergraduate IEEE conferences like FIT, ICACS, or IBCAST."

---

## Slide 9 (9:30 - 10:00) — Conclusion & Q&A
**Say:**
> "To conclude: this project shows that simple, well-tuned models can match modern deep learning on small clinical datasets — *and* with full explainability. The framework is open-source, modular, and aligned with SDG 3, Good Health and Well-being. Thank you — I'm happy to answer any questions."

---

# Possible Questions & Answers

**Q: Why is Logistic Regression the best, not Neural Network?**
> "On small tabular datasets (303 patients), simple models generalize better and overfit less. Neural networks need much more data to outperform. This is also why XGBoost — usually the gold standard — didn't win here."

**Q: What is SHAP exactly?**
> "SHAP comes from cooperative game theory — specifically Shapley values. It treats every feature as a 'player' that contributes to the final prediction. SHAP calculates how much each feature pushed the prediction up or down compared to the average."

**Q: Why didn't you use deep learning?**
> "I did include a Neural Network (MLP). Deep learning shines on images and text. On structured tabular data like clinical records, gradient boosting and SVM are usually equal or better — and far more interpretable."

**Q: How does SMOTE work?**
> "SMOTE generates synthetic minority-class samples by interpolating between existing minority samples and their nearest neighbors. It's better than simply duplicating samples because it adds variety without copying."

**Q: Is this dataset large enough?**
> "303 patients is small but it's the gold-standard benchmark in heart disease ML research, used in over 500 papers. For deployment in a real hospital, I'd validate on a larger cohort like MIMIC-IV with thousands of patients."

**Q: What's the difference between accuracy and ROC-AUC?**
> "Accuracy depends on a fixed threshold (usually 0.5). ROC-AUC measures the model across all possible thresholds, so it tells you the model's *ranking* ability. In healthcare, where the decision threshold can be tuned by the clinician, AUC is the more reliable metric."

**Q: Why 10-fold CV instead of train/test split alone?**
> "10-fold CV gives 10 separate estimates of model performance, which is much more reliable than a single split. It also reduces variance and helps detect overfitting during hyperparameter search."
