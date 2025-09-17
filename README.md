# Telecom Churn Prediction — Case Study

**Author:** Sunil Kumar HN  
**Role:** Data Science Consultant (Internship project)  
**Tools:** Python, Pandas, scikit-learn, XGBoost, Imbalanced-learn, Matplotlib/Seaborn, Power BI, Jupyter Notebook

---

## 1. Project Overview
Telecom operators lose revenue to customer churn. This project builds an end-to-end pipeline to predict churn and provide actionable retention strategies. It includes data cleaning, EDA, feature engineering, model training, evaluation, and a Power BI dashboard for stakeholder communication.

**Key outcome (example):** XGBoost model achieving **~82% accuracy**, precision **~80%**, recall **~78%** on test data (see Results).

---

## 2. Business Problem
- **Problem:** Identify customers likely to churn so the business can run targeted retention campaigns.
- **Business objective:** Reduce monthly churn rate by identifying high-risk customers and recommending interventions (e.g., discounted annual contracts, targeted offers).

---

## 3. Dataset
- Source: Public telecom churn dataset or company dataset — replace with the actual source/link you used.  
- Typical size: 5k–100k rows depending on source.
- Typical columns (example):
  - `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - `tenure`, `PhoneService`, `MultipleLines`, `InternetService`
  - `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`
  - `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`
  - `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn` (Yes/No)

> **Note:** If using private datasets, do **not** upload PII to public GitHub. Provide a sample or synthetic CSV instead and include instructions to replace with private data.

---

## 4. Repo file structure (recommended)
```
telecom-churn-prediction/
├─ data/
│  ├─ raw/
│  │  └─ telecom_churn_sample.csv
│  └─ processed/
│     └─ train.csv
├─ notebooks/
│  ├─ 01_data_cleaning.ipynb
│  ├─ 02_eda.ipynb
│  └─ 03_modeling.ipynb
├─ src/
│  ├─ data_processing.py
│  ├─ features.py
│  ├─ model.py
│  └─ utils.py
├─ images/
│  └─ churn_by_contract.png
├─ dashboard/
│  └─ powerbi_dashboard.pbix
├─ requirements.txt
├─ environment.yml
├─ README.md
└─ LICENSE
```

---

## 5. How to reproduce (local)
1. Clone repo: `git clone https://github.com/<username>/telecom-churn-prediction.git`
2. Create virtual env:
   - using venv:  
     `python -m venv venv && source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
   - or conda: `conda env create -f environment.yml && conda activate telecom-churn`
3. Install: `pip install -r requirements.txt`
4. Run notebooks in order: `01_data_cleaning.ipynb` → `02_eda.ipynb` → `03_modeling.ipynb`

---

## 6. Key steps (summary)

### A. Data cleaning & EDA
- Handle missing values (`TotalCharges` often has blanks), convert types.
- Visualize churn by `Contract`, `tenure`, and `MonthlyCharges`.
- Create histograms, boxplots, correlation heatmap, and churn rate by categorical variables.

### B. Feature engineering
- One-hot encode categorical variables (or use target/ordinal encoding where appropriate).
- Create `tenure_group` buckets (0–12, 12–24, 24–48, 48+).
- Aggregate features if you have transactions (e.g., avg monthly usage).
- Scale numeric features for distance-based models.

### C. Imbalance handling
- Check class distribution (`Churn` often < 30%).
- Use resampling (SMOTE) or class-weighted models.

### D. Modeling
- Baseline: Logistic Regression (interpretable).  
- Tree-based: RandomForest, XGBoost (often best for tabular).  
- Use cross-validation and GridSearch/RandomizedSearch for hyperparameters.

### E. Evaluation
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
- Use a confusion matrix to analyze false positives/negatives.
- Prefer business metric: e.g., identify top 10–20% of customers by predicted churn risk and measure precision@k.

### F. Deployment / Sharing
- Save model with `joblib` or `pickle`.
- Create a simple Streamlit / Flask app to demo predictions.
- Build a Power BI dashboard with key KPIs and model score segments.

---

## 7. Example code snippets

**Load & quick clean**
```python
import pandas as pd
df = pd.read_csv('data/raw/telecom_churn_sample.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['tenure'].fillna(0, inplace=True)
df.dropna(subset=['TotalCharges'], inplace=True)
```

**One hot & train/test split**
```python
from sklearn.model_selection import train_test_split
df = pd.get_dummies(df, drop_first=True)
X = df.drop(['Churn_Yes','customerID'], axis=1)
y = df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

**SMOTE + XGBoost training**
```python
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_res, y_res)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
```

**Save model**
```python
import joblib
joblib.dump(model, 'models/xgb_churn_model.pkl')
```

---

## 8. Evaluation & Results (example)
- **Accuracy:** 82%  
- **Precision:** 80%  
- **Recall:** 78%  
- **ROC-AUC:** 0.86

Interpretation:
- The model correctly identifies a large portion of churners (good recall) while maintaining decent precision.
- Focus on top-scoring customers for retention campaigns reduces wasted offers.

---

## 9. Business Recommendations
1. **Targeted Retention Campaigns:** Offer bundled discounts to high-risk month-to-month customers.  
2. **Incentivize long-term contracts:** Provide attractive offers to customers likely to churn to switch from month-to-month to annual.  
3. **Personalized offers:** Use features like `StreamingMovies`, `TechSupport` to craft product-specific offers.  
4. **Monitor model predictions:** Integrate model scores into CRM to trigger retention flows.

---

## 10. Limitations & Next Steps
- Model depends on historical data quality; consider adding behavioral/usage logs for better performance.
- A/B test recommended retention offers to measure ROI.
- Explore causal analysis to understand drivers of churn versus correlation.

---

## 11. Files & Notebooks
- `notebooks/01_data_cleaning.ipynb` — cleaning + initial EDA  
- `notebooks/02_eda.ipynb` — plots and insights  
- `notebooks/03_modeling.ipynb` — training, CV, and evaluation  
- `src/` — helper functions and reproducible pipeline  
- `dashboard/` — Power BI file or link

---

## 12. References
- [Kaggle Telecom Churn Dataset] (replace with link used)
- scikit-learn, XGBoost docs

---

## 13. Contact
Sunil Kumar HN — hnsunil03@gmail.com — LinkedIn: https://www.linkedin.com/in/sunilkumarhn03/