"""
Trains an attrition prediction model and computes SHAP explanations.

Steps:
  1. Prepare features (encode categoricals, select columns)
  2. Train/test split (stratified, since attrition is imbalanced ~20%)
  3. Baseline: Logistic Regression
  4. Main model: Random Forest (handles nonlinearity, feature interactions)
  5. Evaluate both with precision/recall/F1/ROC-AUC (NOT just accuracy --
     with 80/20 class imbalance, a model that always predicts "No" would
     already be 80% "accurate" and completely useless)
  6. Compute SHAP values for the Random Forest -- this powers the
     "why did the model predict this?" explanation in the web app
  7. Save: trained model, encoders, feature list, SHAP explainer

Outputs (all saved to models/):
  - rf_model.pkl
  - feature_columns.json
  - label_encoders.pkl
  - model_metrics.json
"""
import json
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
import shap

os.makedirs("models", exist_ok=True)

print("=== Loading cleaned data ===")
df = pd.read_csv("data/processed/hr_clean_merged.csv")
df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)

# ---------------------------------------------------------------
# 1. Feature selection & encoding
# ---------------------------------------------------------------
# Categorical columns to encode
cat_cols = ["Gender", "MaritalStatus", "Department", "JobRole", "EducationField",
            "BusinessTravel", "OverTime"]
# Numeric columns to use as-is
num_cols = ["Age", "DistanceFromHome", "Education", "JobLevel", "YearsAtCompany",
            "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
            "NumCompaniesWorked", "TotalWorkingYears", "ManagerChangesLast2Yrs",
            "MonthlyIncome", "PercentSalaryHike", "StockOptionLevel",
            "EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
            "WorkLifeBalance", "JobInvolvement", "PerformanceRating",
            "TrainingTimesLastYear"]

model_df = df[cat_cols + num_cols + ["AttritionFlag"]].copy()

# Safety-net imputation: clean_and_merge.py handled the columns we explicitly
# checked, but any other column with leftover NaNs (e.g. from the 51 employees
# missing a Payroll record) would break sklearn, which can't handle NaN natively.
na_counts = model_df[num_cols].isna().sum()
na_counts = na_counts[na_counts > 0]
if len(na_counts) > 0:
    print(f"\nFound additional missing values not caught by clean_and_merge.py:")
    print(na_counts)
    for col in na_counts.index:
        model_df[col] = model_df[col].fillna(model_df[col].median())
    print("Imputed with column medians as a safety net.")

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

feature_cols = cat_cols + num_cols
X = model_df[feature_cols]
y = model_df["AttritionFlag"]

print(f"Features: {len(feature_cols)} | Samples: {len(X)} | "
      f"Attrition rate: {y.mean():.1%}")

# ---------------------------------------------------------------
# 2. Train/test split (stratified to preserve class balance)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ---------------------------------------------------------------
# 3. Baseline: Logistic Regression
# ---------------------------------------------------------------
print("\n=== Baseline: Logistic Regression ===")
logreg = LogisticRegression(max_iter=5000, class_weight="balanced")
logreg.fit(X_train, y_train)
lr_pred = logreg.predict(X_test)
lr_proba = logreg.predict_proba(X_test)[:, 1]

def evaluate(name, y_true, y_pred, y_proba):
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred), 3),
        "recall": round(recall_score(y_true, y_pred), 3),
        "f1": round(f1_score(y_true, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 3),
    }
    print(f"{name}: {metrics}")
    print(confusion_matrix(y_true, y_pred))
    return metrics

lr_metrics = evaluate("Logistic Regression", y_test, lr_pred, lr_proba)

# ---------------------------------------------------------------
# 4. Main model: Random Forest
# ---------------------------------------------------------------
print("\n=== Main model: Random Forest ===")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

rf_metrics = evaluate("Random Forest", y_test, rf_pred, rf_proba)

print("\nFull classification report (Random Forest):")
print(classification_report(y_test, rf_pred, target_names=["Stayed", "Left"]))

# Feature importance (quick sanity check against our earlier logistic regression findings)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 features by importance:")
print(importances.head(10))

# ---------------------------------------------------------------
# 5. SHAP explainability
# ---------------------------------------------------------------
print("\n=== Computing SHAP values ===")
explainer = shap.TreeExplainer(rf)
# Use a sample of the test set for speed; the app will compute per-employee live
shap_sample = X_test.sample(min(200, len(X_test)), random_state=42)
shap_values = explainer.shap_values(shap_sample)
print("SHAP explainer built successfully.")

# Global feature importance from SHAP (mean absolute value)
if isinstance(shap_values, list):
    sv = shap_values[1]  # class 1 = "Left"
else:
    sv = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols).sort_values(ascending=False)
print("\nTop 10 features by mean |SHAP value| (global importance):")
print(mean_abs_shap.head(10))

# ---------------------------------------------------------------
# 6. Save everything the Streamlit app needs
# ---------------------------------------------------------------
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("models/feature_columns.json", "w") as f:
    json.dump({"cat_cols": cat_cols, "num_cols": num_cols,
               "feature_cols": feature_cols}, f, indent=2)

with open("models/model_metrics.json", "w") as f:
    json.dump({"logistic_regression": lr_metrics, "random_forest": rf_metrics}, f, indent=2)

# Save category options for each encoded column (so the app can build dropdowns)
cat_options = {col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in cat_cols}
with open("models/category_options.json", "w") as f:
    json.dump(cat_options, f, indent=2)

# Save numeric column ranges (so the app can set sensible slider min/max/default)
num_ranges = {col: {"min": float(df[col].min()), "max": float(df[col].max()),
                     "default": float(df[col].median())} for col in num_cols}
with open("models/numeric_ranges.json", "w") as f:
    json.dump(num_ranges, f, indent=2)

print("\n=== DONE ===")
print("Saved: models/rf_model.pkl, label_encoders.pkl, feature_columns.json, "
      "model_metrics.json, category_options.json, numeric_ranges.json")
print(f"\nFinal model performance -- Random Forest ROC-AUC: {rf_metrics['roc_auc']}, "
      f"Recall (catching actual leavers): {rf_metrics['recall']}")
