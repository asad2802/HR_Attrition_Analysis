"""
Splits the clean master dataset into 3 messy source tables that simulate
pulling data from 3 different real-world HR systems:
  1. hris_demographics.csv   - HR Information System (employee master data)
  2. payroll_performance.csv - Payroll/Performance system (comp + surveys)
  3. tenure_manager.csv      - Talent/Org system (tenure, manager history)

Deliberately injects realistic messiness:
  - Different ID formats per system (needs cleaning to join)
  - Missing values
  - Duplicate rows
  - Inconsistent category text/casing
  - A few orphan/mismatched records
"""
import numpy as np
import pandas as pd

np.random.seed(7)
df = pd.read_csv("data/processed/hr_master_clean.csv")
# ---------- 1. HRIS demographics (uses "E100001" style ID, clean-ish) ----------
hris = df[["EmployeeID", "Age", "Gender", "MaritalStatus", "Department",
           "JobRole", "DistanceFromHome", "Education", "EducationField",
           "BusinessTravel", "Attrition"]].copy()

# messiness: inconsistent department text casing/spacing
hris["Department"] = hris["Department"].sample(frac=1.0, random_state=1).reset_index(drop=True)
messy_dept_map = {
    "Sales": ["Sales", "sales ", "SALES"],
    "Research & Development": ["Research & Development", "R&D", "research & development "],
    "Human Resources": ["Human Resources", "HR", "human resources"],
}
hris["Department"] = df["Department"]  # restore correct mapping first
hris["Department"] = hris["Department"].apply(lambda d: np.random.choice(messy_dept_map[d]))

# messiness: some missing Gender / EducationField
mask = np.random.rand(len(hris)) < 0.03
hris.loc[mask, "Gender"] = np.nan
mask2 = np.random.rand(len(hris)) < 0.04
hris.loc[mask2, "EducationField"] = np.nan

# messiness: duplicate a handful of rows (system export glitch)
dupes = hris.sample(15, random_state=2)
hris = pd.concat([hris, dupes], ignore_index=True)

hris.to_csv("data/raw/hris_demographics.csv", index=False)

# ---------- 2. Payroll & performance system (uses "EMP-100001" style ID) --------
payroll = df[["EmployeeID", "MonthlyIncome", "PercentSalaryHike", "StockOptionLevel",
              "PerformanceRating", "JobSatisfaction", "EnvironmentSatisfaction",
              "WorkLifeBalance", "JobInvolvement", "RelationshipSatisfaction",
              "OverTime"]].copy()

# different ID scheme: "EMP-100001" instead of "E100001"
payroll["EmployeeID"] = payroll["EmployeeID"].str.replace("E", "EMP-", regex=False)

# messiness: OverTime as inconsistent Y/N text
payroll["OverTime"] = payroll["OverTime"].map({"Yes": np.random.choice(["Yes", "Y", "yes"]),
                                                 "No": np.random.choice(["No", "N", "no"])})
# simpler + more realistic: randomize per row
payroll["OverTime"] = df["OverTime"].apply(
    lambda v: np.random.choice(["Yes", "Y", "yes", "TRUE"]) if v == "Yes"
    else np.random.choice(["No", "N", "no", "FALSE"]))

# messiness: a few missing MonthlyIncome (payroll export gaps)
mask3 = np.random.rand(len(payroll)) < 0.02
payroll.loc[mask3, "MonthlyIncome"] = np.nan

# messiness: drop 20 random employees entirely from this system (not yet synced)
payroll = payroll.drop(payroll.sample(20, random_state=3).index)

payroll.to_csv("data/raw/payroll_performance.csv", index=False)

# ---------- 3. Tenure & manager history system (uses numeric-only ID) -----------
tenure = df[["EmployeeID", "JobLevel", "YearsAtCompany", "YearsInCurrentRole",
             "YearsSinceLastPromotion", "YearsWithCurrManager",
             "ManagerChangesLast2Yrs", "NumCompaniesWorked", "TotalWorkingYears",
             "TrainingTimesLastYear"]].copy()

# different ID scheme: strips the "E" prefix, just the number as string
tenure["EmployeeID"] = tenure["EmployeeID"].str.replace("E", "", regex=False)

# messiness: a few negative/impossible values from a legacy bug (needs cleaning)
glitch_idx = tenure.sample(8, random_state=4).index
tenure.loc[glitch_idx, "YearsSinceLastPromotion"] = -1

# messiness: missing ManagerChangesLast2Yrs for some
mask4 = np.random.rand(len(tenure)) < 0.03
tenure.loc[mask4, "ManagerChangesLast2Yrs"] = np.nan

tenure.to_csv("data/raw/tenure_manager.csv", index=False)
print("hris_demographics:", hris.shape)
print("payroll_performance:", payroll.shape)
print("tenure_manager:", tenure.shape)
print("\nSample IDs per system (note different formats -> needs cleaning to join):")
print("HRIS:", hris['EmployeeID'].iloc[0])
print("Payroll:", payroll['EmployeeID'].iloc[0])
print("Tenure:", tenure['EmployeeID'].iloc[0])
