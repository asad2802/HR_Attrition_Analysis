"""
Generates a realistic synthetic HR employee dataset with genuine statistical
relationships driving attrition (overtime, tenure, pay, manager changes, satisfaction).
Then splits it into 3 separate "messy" source tables to simulate merging data
from different HR systems (HRIS, Payroll, Performance) - mismatched IDs,
missing values, duplicates, inconsistent formats.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

departments = ["Sales", "Research & Development", "Human Resources"]
dept_weights = [0.30, 0.60, 0.10]

job_roles = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": ["Research Scientist", "Laboratory Technician",
                                "Manufacturing Director", "Healthcare Representative", "Manager"],
    "Human Resources": ["Human Resources", "Manager"],
}

education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree",
                     "Human Resources", "Other"]

df = pd.DataFrame({
    "EmployeeID": [f"E{100000+i}" for i in range(N)],
    "Age": np.random.randint(18, 61, N),
    "Gender": np.random.choice(["Male", "Female"], N, p=[0.6, 0.4]),
    "MaritalStatus": np.random.choice(["Single", "Married", "Divorced"], N, p=[0.32, 0.46, 0.22]),
    "Department": np.random.choice(departments, N, p=dept_weights),
    "DistanceFromHome": np.random.randint(1, 30, N),
    "Education": np.random.randint(1, 6, N),
    "EducationField": np.random.choice(education_fields, N),
})

df["JobRole"] = df["Department"].apply(lambda d: np.random.choice(job_roles[d]))
df["JobLevel"] = np.random.randint(1, 6, N)

# Tenure-related fields
df["YearsAtCompany"] = np.random.exponential(scale=5, size=N).astype(int).clip(0, 40)
df["YearsInCurrentRole"] = (df["YearsAtCompany"] * np.random.uniform(0.2, 1.0, N)).astype(int)
df["YearsSinceLastPromotion"] = (df["YearsAtCompany"] * np.random.uniform(0, 0.6, N)).astype(int)
df["YearsWithCurrManager"] = (df["YearsAtCompany"] * np.random.uniform(0.1, 1.0, N)).astype(int)
df["NumCompaniesWorked"] = np.random.randint(0, 10, N)
df["TotalWorkingYears"] = (df["YearsAtCompany"] + np.random.randint(0, 15, N)).clip(0, 40)

# Manager changes - more changes = higher flight risk
df["ManagerChangesLast2Yrs"] = np.random.poisson(0.6, N)

# Pay
base_pay = 2500 + df["JobLevel"] * 3200 + df["TotalWorkingYears"] * 120
df["MonthlyIncome"] = (base_pay * np.random.uniform(0.85, 1.15, N)).astype(int)
df["PercentSalaryHike"] = np.random.randint(11, 26, N)
df["StockOptionLevel"] = np.random.randint(0, 4, N)

# Work conditions
df["OverTime"] = np.random.choice(["Yes", "No"], N, p=[0.28, 0.72])
df["BusinessTravel"] = np.random.choice(
    ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], N, p=[0.1, 0.71, 0.19])

# Satisfaction survey fields (1-4 ordinal)
df["EnvironmentSatisfaction"] = np.random.randint(1, 5, N)
df["JobSatisfaction"] = np.random.randint(1, 5, N)
df["RelationshipSatisfaction"] = np.random.randint(1, 5, N)
df["WorkLifeBalance"] = np.random.randint(1, 5, N)
df["JobInvolvement"] = np.random.randint(1, 5, N)
df["PerformanceRating"] = np.random.choice([3, 4], N, p=[0.85, 0.15])
df["TrainingTimesLastYear"] = np.random.randint(0, 7, N)

# ---- Build attrition probability from a realistic logistic combination ----
z = (
    -2.6
    + 1.15 * (df["OverTime"] == "Yes")
    + 0.55 * (df["BusinessTravel"] == "Travel_Frequently")
    - 0.09 * df["YearsAtCompany"].clip(0, 15)
    - 0.35 * (df["JobSatisfaction"] - 2.5)
    - 0.30 * (df["WorkLifeBalance"] - 2.5)
    - 0.25 * (df["EnvironmentSatisfaction"] - 2.5)
    + 0.30 * df["ManagerChangesLast2Yrs"]
    - 0.00009 * (df["MonthlyIncome"] - df["MonthlyIncome"].mean())
    + 0.45 * (df["MaritalStatus"] == "Single")
    - 0.03 * (df["Age"] - 37)
    + 0.12 * df["NumCompaniesWorked"].clip(0, 6)
    + np.random.normal(0, 0.6, N)  # noise
)
prob = 1 / (1 + np.exp(-z))
df["Attrition"] = np.where(np.random.uniform(0, 1, N) < prob, "Yes", "No")

print("Attrition rate:", (df["Attrition"] == "Yes").mean().round(3))
print(df.groupby("OverTime")["Attrition"].apply(lambda s: (s == "Yes").mean()).round(3))

df.to_csv("data/processed/hr_master_clean.csv", index=False)
print("Saved master dataset:", df.shape)
