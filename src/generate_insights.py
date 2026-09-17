"""
Reads the outputs of eda_stats.py (segment rates + logistic regression odds ratios)
and automatically generates a plain-English findings & recommendations report.

This is rule-based (not an LLM call) so it's fast, free, and fully reproducible --
the logic just translates statistical thresholds into narrative language and maps
each significant risk factor to a concrete HR action.

Output: reports/attrition_insights_report.md
"""
import pandas as pd

eda = pd.read_csv("data/processed/eda_summary.csv")
odds = pd.read_csv("data/processed/logistic_regression_odds_ratios.csv", index_col=0)
odds = odds.drop("Intercept", errors="ignore")

overall_rate = pd.read_csv("data/processed/hr_clean_merged.csv")
overall_rate = (overall_rate["Attrition"] == "Yes").mean()

# ---------------------------------------------------------------
# Recommended actions mapped to each modeled factor.
# Only fires if that factor turned out significant (p < 0.05) AND
# meaningfully increases risk (OR > 1.15) or protects (OR < 0.9).
# ---------------------------------------------------------------
ACTION_MAP = {
    "OverTime_bin": {
        "risk_label": "Overtime work",
        "action": "Audit workload distribution in teams with high overtime. Review "
                  "whether overtime is fairly compensated, and consider mandatory "
                  "recovery time after sustained high-overtime periods.",
    },
    "FreqTravel_bin": {
        "risk_label": "Frequent business travel",
        "action": "Reassess assignment of frequent-travel roles. Consider rotating "
                  "travel-heavy responsibilities across the team or increasing "
                  "travel-related support/compensation.",
    },
    "ManagerChangesLast2Yrs": {
        "risk_label": "Frequent manager changes",
        "action": "Build a structured manager-transition process (e.g. 30/60/90-day "
                  "check-ins after a manager change) to stabilize the employee-manager "
                  "relationship during handoffs.",
    },
    "Single_bin": {
        "risk_label": "Marital status (single employees)",
        "action": "CAUTION: this is a demographic correlation, not a lever to act on "
                  "directly (acting on marital status would be inappropriate/possibly "
                  "discriminatory). Investigate underlying drivers instead -- e.g. "
                  "belonging/social-connection programs -- rather than targeting by "
                  "marital status.",
    },
    "YearsAtCompany": {
        "risk_label": "Each additional year at the company",
        "action": "Prioritize retention efforts in the first 1-3 years specifically "
                  "(see tenure-band table below) -- structured onboarding, early "
                  "mentorship, and a 90-day and 1-year check-in.",
        "protective": True,
    },
    "JobSatisfaction": {
        "risk_label": "Job satisfaction",
        "action": "Run regular pulse surveys and act on the results visibly. Higher "
                  "satisfaction is one of the strongest protective factors in the model.",
        "protective": True,
    },
    "WorkLifeBalance": {
        "risk_label": "Work-life balance score",
        "action": "Review flexible-work and PTO policies, especially in teams that "
                  "score low on this metric.",
        "protective": True,
    },
    "EnvironmentSatisfaction": {
        "risk_label": "Environment satisfaction score",
        "action": "Investigate team/workplace environment factors (management style, "
                  "physical/remote setup, team dynamics) in low-scoring departments.",
        "protective": True,
    },
    "MonthlyIncome": {
        "risk_label": "Monthly income",
        "action": "Conduct a pay-equity review, prioritizing the lowest pay quartile "
                  "(see pay-band table below), which shows the highest attrition rate.",
        "protective": True,
    },
    "Age": {
        "risk_label": "Age",
        "action": "No direct action recommended (age is not an appropriate lever) -- "
                  "this likely reflects early-career/early-tenure dynamics already "
                  "captured by the tenure recommendation above.",
        "protective": True,
    },
}

lines = []
lines.append("# HR Attrition: Key Findings & Recommendations")
lines.append(f"\n_Auto-generated from `eda_stats.py` output. "
              f"Overall attrition rate: **{overall_rate:.1%}**_\n")

# ---------------------------------------------------------------
# Section 1: Key findings (ranked by effect size)
# ---------------------------------------------------------------
lines.append("## Key Findings\n")

sig = odds[odds["p_value"] < 0.05].copy()
sig["abs_effect"] = sig["Odds_Ratio"].apply(lambda x: max(x, 1 / x))  # distance from 1
sig = sig.sort_values("abs_effect", ascending=False)

finding_num = 1
for factor, row in sig.iterrows():
    or_val = row["Odds_Ratio"]
    if factor not in ACTION_MAP:
        continue
    label = ACTION_MAP[factor]["risk_label"]
    if or_val > 1.15:
        direction = f"increases attrition odds by **{(or_val - 1) * 100:.0f}%**"
    elif or_val < 0.9:
        direction = f"reduces attrition odds by **{(1 - or_val) * 100:.0f}%** per unit increase"
    else:
        continue  # too small an effect to call out
    lines.append(f"{finding_num}. **{label}** {direction} "
                  f"(odds ratio = {or_val:.2f}, p = {row['p_value']:.4f}), "
                  f"holding other factors constant.")
    finding_num += 1

# ---------------------------------------------------------------
# Section 2: Highest-risk segments (from EDA)
# ---------------------------------------------------------------
lines.append("\n## Highest-Risk Segments\n")
for seg_name in ["TenureBand", "PayBand", "OverTime"]:
    seg_data = eda[eda["segment"] == seg_name].sort_values("attrition_rate", ascending=False)
    if seg_data.empty:
        continue
    top = seg_data.iloc[0]
    lines.append(f"- **{seg_name}**: highest risk group is `{top['value']}` "
                  f"at **{top['attrition_rate']:.1%}** attrition (n={int(top['n'])}), "
                  f"vs overall average of {overall_rate:.1%}.")

# ---------------------------------------------------------------
# Section 3: Recommended actions
# ---------------------------------------------------------------
lines.append("\n## Recommended Actions\n")
action_num = 1
seen_actions = set()
for factor in sig.index:
    if factor not in ACTION_MAP:
        continue
    action_text = ACTION_MAP[factor]["action"]
    if action_text in seen_actions:
        continue
    seen_actions.add(action_text)
    lines.append(f"{action_num}. {action_text}")
    action_num += 1

lines.append("\n## Caveats\n")
lines.append("- This dataset is synthetic/for portfolio purposes; findings illustrate "
              "the *method*, not real organizational conclusions.")
lines.append("- Odds ratios show association, not proven causation -- e.g. low "
              "satisfaction may be a symptom of an employee already planning to leave, "
              "not only a cause.")
lines.append("- Any action involving protected characteristics (marital status, age, "
              "gender) should be reviewed for legal/ethical compliance before use -- "
              "flagged findings here are for awareness, not direct targeting.")

report = "\n".join(lines)
with open("reports/attrition_insights_report.md", "w") as f:
    f.write(report)

print(report)
print("\n\nSaved to reports/attrition_insights_report.md")
