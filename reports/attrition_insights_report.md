# HR Attrition: Key Findings & Recommendations

_Auto-generated from `eda_stats.py` output. Overall attrition rate: **20.3%**_

## Key Findings

1. **Overtime work** increases attrition odds by **182%** (odds ratio = 2.82, p = 0.0000), holding other factors constant.
2. **Frequent business travel** increases attrition odds by **71%** (odds ratio = 1.71, p = 0.0016), holding other factors constant.
3. **Marital status (single employees)** increases attrition odds by **57%** (odds ratio = 1.57, p = 0.0022), holding other factors constant.
4. **Frequent manager changes** increases attrition odds by **41%** (odds ratio = 1.41, p = 0.0000), holding other factors constant.
5. **Work-life balance score** reduces attrition odds by **25%** per unit increase (odds ratio = 0.75, p = 0.0000), holding other factors constant.
6. **Job satisfaction** reduces attrition odds by **23%** per unit increase (odds ratio = 0.77, p = 0.0000), holding other factors constant.
7. **Environment satisfaction score** reduces attrition odds by **23%** per unit increase (odds ratio = 0.77, p = 0.0001), holding other factors constant.
8. **Each additional year at the company** reduces attrition odds by **12%** per unit increase (odds ratio = 0.88, p = 0.0000), holding other factors constant.

## Highest-Risk Segments

- **TenureBand**: highest risk group is `0-1 yr` at **27.0%** attrition (n=488), vs overall average of 20.3%.
- **PayBand**: highest risk group is `Q1 (lowest)` at **27.2%** attrition (n=375), vs overall average of 20.3%.
- **OverTime**: highest risk group is `Yes` at **31.7%** attrition (n=404), vs overall average of 20.3%.

## Recommended Actions

1. Audit workload distribution in teams with high overtime. Review whether overtime is fairly compensated, and consider mandatory recovery time after sustained high-overtime periods.
2. Reassess assignment of frequent-travel roles. Consider rotating travel-heavy responsibilities across the team or increasing travel-related support/compensation.
3. CAUTION: this is a demographic correlation, not a lever to act on directly (acting on marital status would be inappropriate/possibly discriminatory). Investigate underlying drivers instead -- e.g. belonging/social-connection programs -- rather than targeting by marital status.
4. Build a structured manager-transition process (e.g. 30/60/90-day check-ins after a manager change) to stabilize the employee-manager relationship during handoffs.
5. Review flexible-work and PTO policies, especially in teams that score low on this metric.
6. Run regular pulse surveys and act on the results visibly. Higher satisfaction is one of the strongest protective factors in the model.
7. Investigate team/workplace environment factors (management style, physical/remote setup, team dynamics) in low-scoring departments.
8. Prioritize retention efforts in the first 1-3 years specifically (see tenure-band table below) -- structured onboarding, early mentorship, and a 90-day and 1-year check-in.
9. No direct action recommended (age is not an appropriate lever) -- this likely reflects early-career/early-tenure dynamics already captured by the tenure recommendation above.
10. Conduct a pay-equity review, prioritizing the lowest pay quartile (see pay-band table below), which shows the highest attrition rate.

## Caveats

- This dataset is synthetic/for portfolio purposes; findings illustrate the *method*, not real organizational conclusions.
- Odds ratios show association, not proven causation -- e.g. low satisfaction may be a symptom of an employee already planning to leave, not only a cause.
- Any action involving protected characteristics (marital status, age, gender) should be reviewed for legal/ethical compliance before use -- flagged findings here are for awareness, not direct targeting.