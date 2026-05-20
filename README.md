# Telecom Customer Churn Prediction

## 📋 Project Overview

This repository contains the code and supporting materials for a complete end-to-end machine learning investigation into **customer churn prediction** in the telecommunications industry. The project applies exploratory data analysis, data cleaning, feature engineering, and four classification algorithms to predict whether a customer will cancel their subscription.

**Research Question:** Can machine learning models accurately predict customer churn in a telecommunications context, and which features and algorithms best support a proactive retention strategy?

---

## 📂 Repository Structure

```
├── Telco_churn_analysis.ipynb   # Main Jupyter notebook (all code)
├── Telco-Customer-Churn.csv      # Dataset (see source below)
└── README.md                     # This file
```

---

## 📊 Dataset

### Source
**Telco Customer Churn**

| Detail | Value |
|---|---|
| **Source** | [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Licence** | Public Domain / Open Use |
| **Format** | CSV |

### Overview

The dataset represents **7,043 customers** of a fictitious US telecommunications company. Each row captures a customer's demographics, subscribed services, billing account information, and a binary label indicating whether they churned within the last month.

| Property | Value |
|---|---|
| Rows | 7,043 |
| Columns | 21 |
| Target Variable | `Churn` (Yes / No) |
| Churn Rate | 26.5% (1,869 churned / 5,174 retained) |
| Missing Values | 11 (in `TotalCharges` only, 0.16%) |
| Class Imbalance Ratio | ~2.8 : 1 (No : Yes) |

### Feature Description

The 21 columns span four categories:

#### 🧑 Demographics (4 features)

| Column | Type | Description |
|---|---|---|
| `customerID` | String | Unique customer identifier (dropped before modelling) |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Integer (0/1) | Whether the customer is a senior citizen |
| `Partner` | Categorical | Whether the customer has a partner (Yes/No) |
| `Dependents` | Categorical | Whether the customer has dependents (Yes/No) |

#### 📱 Services Subscribed (8 features)

| Column | Type | Description |
|---|---|---|
| `PhoneService` | Categorical | Whether the customer has a phone service (Yes/No) |
| `MultipleLines` | Categorical | Multiple lines, single line, or no phone service |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Yes / No / No internet service |
| `OnlineBackup` | Categorical | Yes / No / No internet service |
| `DeviceProtection` | Categorical | Yes / No / No internet service |
| `TechSupport` | Categorical | Yes / No / No internet service |
| `StreamingTV` | Categorical | Yes / No / No internet service |
| `StreamingMovies` | Categorical | Yes / No / No internet service |

#### 🧾 Account Information (5 features)

| Column | Type | Description |
|---|---|---|
| `tenure` | Integer | Number of months the customer has been with the company (0–72) |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Whether the customer uses paperless billing (Yes/No) |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Float | Monthly billing amount ($) |
| `TotalCharges` | Float | Total amount charged over tenure ($) — contains 11 missing values |

#### 🎯 Target Variable

| Column | Type | Description |
|---|---|---|
| `Churn` | Categorical | Whether the customer churned in the last month (Yes / No) |

---

## ⚙️ Methodology

### Notebook Structure

| Section | Content |
|---|---|
| **Section 1** — Imports & Setup | Libraries and global settings |
| **Section 2** — EDA, Cleaning & Preprocessing | Dataset inspection, missing values, cleaning, feature engineering, encoding, visualisations (Figures 1 & 2) |
| **Section 3** — Model Development & Tuning | Model selection, baseline architecture, cross-validation setup, hyperparameter tuning |
| **Section 4** — Results & Evaluation | Test set metrics, confusion matrices, ROC curves, feature importance, learning curves (Figures 3 & 4) |

### Preprocessing Pipeline
1. Drop `customerID` (non-informative identifier)
2. Median imputation for 11 missing `TotalCharges` values (median preferred over mean due to right-skewed distribution)
3. Map `SeniorCitizen` 0/1 → No/Yes for consistent one-hot encoding
4. Feature engineering: `AvgMonthlySpend` (`TotalCharges / (tenure + 1)`) and `CustomerSegment` (tenure bands: New 0–12m, Mid-term 13–36m, Loyal 37–72m)
5. One-hot encoding of all categorical features (53 total features after encoding)
6. 80/20 stratified train/test split (random state = 42) — preserves 26.5% churn ratio in both partitions
7. `StandardScaler` fitted on training data only, applied to Logistic Regression inputs only (tree-based models are scale-invariant); prevents data leakage

### Models Evaluated

| Model | Type | Tuning | Best Parameters | CV AUC |
|---|---|---|---|---|
| Logistic Regression | Parametric baseline | Default (`max_iter=1000`) | — | — |
| Decision Tree | Tree baseline | Default parameters | — | — |
| Random Forest | Bagging ensemble | GridSearchCV (5-fold, 240 fits) | `n_estimators=200`, `max_depth=10`, `max_features='log2'`, `min_samples_leaf=2` | 0.8446 ± 0.0098 |
| Gradient Boosting (HistGBM) | Boosting ensemble | GridSearchCV (5-fold, 180 fits) | `max_iter=100`, `learning_rate=0.05`, `max_depth=4`, `l2_regularization=0.0` | 0.8492 ± 0.0108 |

### Key Analytical Findings

**Correlation Analysis (Figure 2):**
- Strongest positive correlates (churn risk): `MonthlyCharges` (r = 0.193), `PaperlessBilling` (r = 0.192), `AvgMonthlySpend` (engineered feature, moderate positive)
- Strongest negative correlates (retention): `Contract` (r = −0.397), `tenure` (r = −0.352)

**Feature Importance — Random Forest (Figure 4):**
Top 5 most predictive features by Mean Decrease in Impurity: `tenure`, `TotalCharges`, `Contract_Month-to-month`, `MonthlyCharges`, `AvgMonthlySpend` — validating the feature engineering step.

### Figures Generated

| Figure | File | Content |
|---|---|---|
| Figure 1 | `fig1_eda.png` | 6-panel EDA: churn distribution, churn by contract type, churn by internet service, tenure/charge distributions by churn, scatter of tenure vs monthly charges |
| Figure 2 | `fig2_correlation.png` | Pearson correlation of all features with the binary Churn target |
| Figure 3 | `fig3_evaluation.png` | Confusion matrices for all 4 models, overlaid ROC curves, grouped metric bar chart |
| Figure 4 | `fig4_importance_learning.png` | Top 15 feature importances (Random Forest) and learning curves (Gradient Boosting) |

### Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting (tuned)** | **0.8062** | **0.6747** | 0.5214 | 0.5882 | **0.8460** |
| Logistic Regression | 0.8070 | 0.6689 | **0.5401** | **0.5976** | 0.8458 |
| Random Forest (tuned) | 0.7970 | 0.6528 | 0.5027 | 0.5680 | 0.8423 |
| Decision Tree | 0.7204 | 0.4751 | 0.5107 | 0.4923 | 0.6532 |

> Primary metric: ROC-AUC (suitable for imbalanced binary classification)

---

## 🛠️ How to Run

### Option 1 — Google Colab (Recommended)
1. Open `Telco_churn_analysis.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `Telco-Customer-Churn.csv` when prompted in Section 2
3. Run all cells in order (`Runtime → Run all`)

### Option 2 — Local Jupyter
```bash
# Clone the repository
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Launch Jupyter
jupyter notebook Telco_churn_analysis.ipynb
```

> **Note:** Update `DATA_PATH` in Section 2 of the notebook to match your local CSV file path if running locally.

---

## 📦 Dependencies

```
Python      >= 3.9
numpy       >= 1.23
pandas      >= 1.5
matplotlib  >= 3.6
seaborn     >= 0.12
scikit-learn>= 1.1
```

---

## ⚠️ Ethical Considerations

- The dataset is **fully anonymised** — no personally identifiable information is present
- Variables such as `gender` and `SeniorCitizen` should be monitored for **proxy discrimination** in any production deployment
- Any real-world deployment must comply with **UK GDPR** and applicable data protection regulations
- Model outputs should support, not replace, human decision-making in customer retention