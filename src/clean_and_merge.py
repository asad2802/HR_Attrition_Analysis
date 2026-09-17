"""
Cleans and merges the 3 messy HR source tables into one analysis-ready dataset.

Handles, in order:
  1. Standardizing 3 different EmployeeID formats into one common key
  2. Removing duplicate rows (HRIS export glitch)
  3. Standardizing inconsistent categorical text (Department, OverTime)
  4. Fixing impossible values (negative YearsSinceLastPromotion)
  5. Handling missing values (documented decision per column)
  6. Merging the 3 tables and handling employees missing from Payroll

Outputs:
  - data/processed/hr_clean_merged.csv   <- final analysis-ready table
  - A printed data-quality log summarizing every fix made (paste into your README)
"""
import numpy as np
import pandas as pd

log = []  # collect data-quality findings to print at the end


def note(msg):
    log.append(msg)
    print("•", msg)


print("=== Loading raw source files ===")
hris = pd.read_csv("data/raw/hris_demographics.csv")
payroll = pd.read_csv("data/raw/payroll_performance.csv")
tenure = pd.read_csv("data/raw/tenure_manager.csv")
note(f"Loaded HRIS ({len(hris)} rows), Payroll ({len(payroll)} rows), "
     f"Tenure ({len(tenure)} rows)")

# ---------------------------------------------------------------
# 1. Standardize EmployeeID formats into one common key: integer ID
#    HRIS:    "E100000"    -> 100000
#    Payroll: "EMP-100000" -> 100000
#    Tenure:  "100000"     -> 100000
# ---------------------------------------------------------------
print("\n=== Step 1: Standardizing ID formats ===")
hris["emp_key"] = hris["EmployeeID"].str.extract(r"(\d+)").astype(int)
payroll["emp_key"] = payroll["EmployeeID"].str.extract(r"(\d+)").astype(int)
tenure["emp_key"] = tenure["EmployeeID"].astype(int)
note("Standardized 3 different ID formats (E-, EMP--, numeric) into a single "
     "integer join key `emp_key`")

# ---------------------------------------------------------------
# 2. Remove duplicate rows in HRIS
# ---------------------------------------------------------------
print("\n=== Step 2: Removing duplicates ===")
before = len(hris)
hris = hris.drop_duplicates(subset="emp_key", keep="first")
note(f"Dropped {before - len(hris)} duplicate rows from HRIS (kept first occurrence)")

# ---------------------------------------------------------------
# 3. Standardize inconsistent categorical text
# ---------------------------------------------------------------
print("\n=== Step 3: Standardizing categorical text ===")
dept_map = {
    "sales": "Sales", "sales ": "Sales", "SALES": "Sales", "Sales": "Sales",
    "Research & Development": "Research & Development", "R&D": "Research & Development",
    "research & development ": "Research & Development",
    "Human Resources": "Human Resources", "HR": "Human Resources",
    "human resources": "Human Resources",
}
hris["Department"] = hris["Department"].str.strip().map(
    lambda x: dept_map.get(x, dept_map.get(x.strip().upper() if isinstance(x, str) else x, x))
)
# safer explicit pass in case of casing edge cases
hris["Department"] = hris["Department"].replace({
    "SALES": "Sales", "sales": "Sales",
    "R&D": "Research & Development", "research & development": "Research & Development",
    "HR": "Human Resources", "human resources": "Human Resources",
})
note(f"Standardized Department text to 3 clean categories: "
     f"{sorted(hris['Department'].dropna().unique())}")

payroll["OverTime"] = payroll["OverTime"].str.strip().str.upper().map(
    lambda x: "Yes" if x in ("YES", "Y", "TRUE") else ("No" if x in ("NO", "N", "FALSE") else x)
)
note("Standardized OverTime values (Yes/Y/yes/TRUE -> 'Yes', No/N/no/FALSE -> 'No')")

# ---------------------------------------------------------------
# 4. Fix impossible values
# ---------------------------------------------------------------
print("\n=== Step 4: Fixing impossible values ===")
bad = (tenure["YearsSinceLastPromotion"] < 0).sum()
tenure.loc[tenure["YearsSinceLastPromotion"] < 0, "YearsSinceLastPromotion"] = np.nan
note(f"Found {bad} rows with impossible negative YearsSinceLastPromotion "
     f"(legacy system bug) -> set to NaN, will impute with median")

# ---------------------------------------------------------------
# 5. Merge tables (left join from HRIS, since it's our most complete source)
# ---------------------------------------------------------------
print("\n=== Step 5: Merging tables ===")
merged = hris.merge(payroll.drop(columns=["EmployeeID"]), on="emp_key", how="left")
merged = merged.merge(tenure.drop(columns=["EmployeeID"]), on="emp_key", how="left")

missing_payroll = merged["MonthlyIncome"].isna().sum()
note(f"{missing_payroll} employees have no matching Payroll record "
     f"(not yet synced to that system) -> flagged via missing values, "
     f"NOT dropped, since we don't want to lose attrition signal from HRIS")

# ---------------------------------------------------------------
# 6. Handle missing values (documented, column-by-column decisions)
# ---------------------------------------------------------------
print("\n=== Step 6: Handling missing values ===")

# Numeric columns -> impute with median, and flag that they were imputed
numeric_impute_cols = ["MonthlyIncome", "YearsSinceLastPromotion", "ManagerChangesLast2Yrs"]
for col in numeric_impute_cols:
    n_missing = merged[col].isna().sum()
    if n_missing > 0:
        merged[f"{col}_was_missing"] = merged[col].isna().astype(int)
        median_val = merged[col].median()
        merged[col] = merged[col].fillna(median_val)
        note(f"Imputed {n_missing} missing '{col}' values with median "
             f"({median_val:.1f}); added '{col}_was_missing' flag column")

# Categorical columns -> impute with "Unknown" (don't invent a category)
categorical_impute_cols = ["Gender", "EducationField"]
for col in categorical_impute_cols:
    n_missing = merged[col].isna().sum()
    if n_missing > 0:
        merged[col] = merged[col].fillna("Unknown")
        note(f"Filled {n_missing} missing '{col}' values with 'Unknown' "
             f"(avoided guessing a value that could bias the model)")

# ---------------------------------------------------------------
# Final cleanup: drop redundant ID columns, reorder, save
# ---------------------------------------------------------------
merged = merged.rename(columns={"emp_key": "EmployeeKey"})
cols = ["EmployeeKey"] + [c for c in merged.columns if c != "EmployeeKey"]
merged = merged[cols]

merged.to_csv("data/processed/hr_clean_merged.csv", index=False)

print(f"\n=== DONE: saved data/processed/hr_clean_merged.csv ===")
print(f"Final shape: {merged.shape}")
print(f"Attrition rate in merged data: {(merged['Attrition']=='Yes').mean():.3f}")

print("\n=== DATA QUALITY LOG (paste this into your README) ===")
for i, item in enumerate(log, 1):
    print(f"{i}. {item}")
