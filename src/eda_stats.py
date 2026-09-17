"""
EDA + statistical testing on the cleaned HR dataset.

Produces:
  - data/processed/eda_summary.csv     <- attrition rate by key segments
  - reports/figures/*.png              <- charts for README/portfolio
  - Printed chi-square test results    <- which categorical factors are
                                           statistically significant drivers
  - Printed logistic regression        <- odds ratios (magnitude + direction
                                           of effect for each factor, holding
                                           others constant)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf

os.makedirs("reports/figures", exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_csv("data/processed/hr_clean_merged.csv")
df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
overall_rate = df["AttritionFlag"].mean()
print(f"Overall attrition rate: {overall_rate:.1%}\n")

# ---------------------------------------------------------------
# 1. Attrition rate by key segments
# ---------------------------------------------------------------
print("=== Attrition rate by segment ===")
segments = ["Department", "OverTime", "BusinessTravel", "MaritalStatus", "Gender"]
summary_rows = []
for seg in segments:
    rates = df.groupby(seg)["AttritionFlag"].agg(["mean", "count"]).round(3)
    rates.columns = ["attrition_rate", "n"]
    print(f"\n-- {seg} --")
    print(rates)
    for idx, row in rates.iterrows():
        summary_rows.append({"segment": seg, "value": idx,
                              "attrition_rate": row["attrition_rate"], "n": int(row["n"])})

# tenure bands
df["TenureBand"] = pd.cut(df["YearsAtCompany"], bins=[-1, 1, 3, 5, 10, 100],
                           labels=["0-1 yr", "2-3 yrs", "4-5 yrs", "6-10 yrs", "10+ yrs"])
tenure_rates = df.groupby("TenureBand", observed=True)["AttritionFlag"].agg(["mean", "count"]).round(3)
print("\n-- TenureBand --")
print(tenure_rates)
for idx, row in tenure_rates.iterrows():
    summary_rows.append({"segment": "TenureBand", "value": str(idx),
                          "attrition_rate": row["mean"], "n": int(row["count"])})

# pay quartiles
df["PayBand"] = pd.qcut(df["MonthlyIncome"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
pay_rates = df.groupby("PayBand", observed=True)["AttritionFlag"].agg(["mean", "count"]).round(3)
print("\n-- PayBand --")
print(pay_rates)
for idx, row in pay_rates.iterrows():
    summary_rows.append({"segment": "PayBand", "value": str(idx),
                          "attrition_rate": row["mean"], "n": int(row["count"])})

pd.DataFrame(summary_rows).to_csv("data/processed/eda_summary.csv", index=False)
print("\nSaved data/processed/eda_summary.csv")

# ---------------------------------------------------------------
# 2. Charts
# ---------------------------------------------------------------
print("\n=== Saving charts to reports/figures/ ===")

fig, ax = plt.subplots(figsize=(7, 4))
df.groupby("Department")["AttritionFlag"].mean().sort_values().plot(
    kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Attrition Rate by Department")
ax.set_xlabel("Attrition Rate")
plt.tight_layout()
plt.savefig("reports/figures/attrition_by_department.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
df.groupby("OverTime")["AttritionFlag"].mean().plot(kind="bar", ax=ax, color="#DD8452")
ax.set_title("Attrition Rate: Overtime vs No Overtime")
ax.set_ylabel("Attrition Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/figures/attrition_by_overtime.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
tenure_rates["mean"].plot(kind="bar", ax=ax, color="#55A868")
ax.set_title("Attrition Rate by Tenure Band")
ax.set_ylabel("Attrition Rate")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("reports/figures/attrition_by_tenure.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
pay_rates["mean"].plot(kind="bar", ax=ax, color="#8172B2")
ax.set_title("Attrition Rate by Pay Quartile")
ax.set_ylabel("Attrition Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/figures/attrition_by_paybandd.png", dpi=150)
plt.close()

print("Saved 4 chart images.")

# ---------------------------------------------------------------
# 3. Chi-square tests (are these categorical differences statistically real?)
# ---------------------------------------------------------------
print("\n=== Chi-square tests: is the segment difference statistically significant? ===")
for seg in ["Department", "OverTime", "BusinessTravel", "MaritalStatus"]:
    table = pd.crosstab(df[seg], df["Attrition"])
    chi2, p, dof, expected = chi2_contingency(table)
    sig = "SIGNIFICANT (p < 0.05)" if p < 0.05 else "not significant"
    print(f"{seg}: chi2={chi2:.2f}, p={p:.4f} -> {sig}")

# ---------------------------------------------------------------
# 4. Logistic regression for odds ratios
# ---------------------------------------------------------------
print("\n=== Logistic regression: odds ratios (holding other factors constant) ===")
model_df = df.copy()
model_df["OverTime_bin"] = (model_df["OverTime"] == "Yes").astype(int)
model_df["Single_bin"] = (model_df["MaritalStatus"] == "Single").astype(int)
model_df["FreqTravel_bin"] = (model_df["BusinessTravel"] == "Travel_Frequently").astype(int)

formula = ("AttritionFlag ~ OverTime_bin + FreqTravel_bin + Single_bin + "
           "YearsAtCompany + JobSatisfaction + WorkLifeBalance + "
           "EnvironmentSatisfaction + ManagerChangesLast2Yrs + MonthlyIncome + Age")

logit_model = smf.logit(formula, data=model_df).fit(disp=0)
print(logit_model.summary())

odds_ratios = np.exp(logit_model.params).round(3)
conf = np.exp(logit_model.conf_int())
conf.columns = ["OR_low_95%", "OR_high_95%"]
or_table = pd.concat([odds_ratios.rename("Odds_Ratio"), conf,
                       logit_model.pvalues.rename("p_value").round(4)], axis=1)
print("\n=== Odds Ratio summary (>1 = increases attrition risk) ===")
print(or_table)
or_table.to_csv("data/processed/logistic_regression_odds_ratios.csv")
print("\nSaved data/processed/logistic_regression_odds_ratios.csv")