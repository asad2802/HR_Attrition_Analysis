"""
HR Attrition Analysis - Streamlit App

Run with: streamlit run app/app.py  (from the project root)

Tabs:
  1. Predict        - enter an employee's details, get a risk score + SHAP explanation
  2. Insights        - EDA charts + auto-generated findings/recommendations
  3. About           - project summary, methodology, caveats
"""
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="HRHELP · Attrition Risk", page_icon="♞", layout="wide")

# ---------------------------------------------------------------
# Brand styling — HRHELP, chess-inspired design system
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
  --ink: #1A1A18;
  --ivory: #EDE6D6;
  --walnut: #8B5E34;
  --walnut-dark: #6B4623;
  --moss: #4C7A5E;
  --amber: #C08A2E;
  --crimson: #7A2E2E;
  --line: #C9BFA8;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }
.stApp { background-color: var(--ivory); }

/* Masthead */
.hrhelp-masthead {
  display: flex; align-items: baseline; gap: 14px; padding-top: 4px;
}
.hrhelp-mark { font-size: 1.8rem; color: var(--walnut); }
.hrhelp-wordmark {
  font-family: 'Playfair Display', serif; font-weight: 700; font-size: 2.5rem;
  color: var(--ink); letter-spacing: -0.01em;
}
.hrhelp-tagline {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #6B5F4D;
}
.checker-strip {
  height: 8px; margin: 14px 0 22px 0;
  background-image:
    linear-gradient(45deg, var(--ink) 25%, transparent 25%, transparent 75%, var(--ink) 75%, var(--ink)),
    linear-gradient(45deg, var(--ink) 25%, transparent 25%, transparent 75%, var(--ink) 75%, var(--ink));
  background-size: 16px 16px; background-position: 0 0, 8px 8px;
  opacity: 0.85;
}

/* Section headings */
h2, h3 { font-family: 'Playfair Display', serif !important; font-weight: 700 !important; color: var(--ink) !important; }

/* Tabs -> underline style */
button[data-baseweb="tab"] {
  font-family: 'IBM Plex Sans', sans-serif; font-weight: 500; font-size: 1rem; color: #6B5F4D;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--ink) !important; border-bottom: 3px solid var(--walnut) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
  background-color: var(--walnut); border: none; border-radius: 2px;
  font-weight: 500; letter-spacing: 0.01em;
}
.stButton > button[kind="primary"]:hover { background-color: var(--walnut-dark); }

/* Risk result card */
.risk-card {
  border: 1px solid var(--line); border-left: 6px solid var(--ink);
  background: #FBFAF6; padding: 22px 26px; border-radius: 2px;
}
.risk-card.mate { border-left-color: var(--crimson); }
.risk-card.check { border-left-color: var(--amber); }
.risk-card.solid { border-left-color: var(--moss); }
.risk-glyph { font-size: 2rem; line-height: 1; margin-bottom: 4px; }
.risk-score {
  font-family: 'IBM Plex Mono', monospace; font-size: 2.6rem; font-weight: 600; line-height: 1;
}
.risk-verdict { font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; margin-top: 6px; }
.risk-verdict.mate { color: var(--crimson); }
.risk-verdict.check { color: var(--amber); }
.risk-verdict.solid { color: var(--moss); }

.shap-row {
  padding: 7px 0; border-bottom: 1px solid var(--line);
  font-size: 0.95rem; font-family: 'IBM Plex Sans', sans-serif;
}
.shap-row b { font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
.shap-up { color: var(--crimson); font-weight: 500; }
.shap-down { color: var(--moss); font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Load everything once, cache it
# ---------------------------------------------------------------
@st.cache_resource
def load_model_assets():
    with open("models/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("models/feature_columns.json") as f:
        cols = json.load(f)
    with open("models/category_options.json") as f:
        cat_options = json.load(f)
    with open("models/numeric_ranges.json") as f:
        num_ranges = json.load(f)
    with open("models/model_metrics.json") as f:
        metrics = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, encoders, cols, cat_options, num_ranges, metrics, explainer


@st.cache_data
def load_eda_summary():
    return pd.read_csv("data/processed/eda_summary.csv")


@st.cache_data
def load_insights_report():
    with open("reports/attrition_insights_report.md") as f:
        return f.read()


model, encoders, cols, cat_options, num_ranges, metrics, explainer = load_model_assets()
cat_cols, num_cols, feature_cols = cols["cat_cols"], cols["num_cols"], cols["feature_cols"]

st.markdown("""
<div class="hrhelp-masthead">
  <span class="hrhelp-mark">♞</span>
  <span class="hrhelp-wordmark">HRHELP</span>
  <span class="hrhelp-tagline">read the board before you lose the piece</span>
</div>
<div class="checker-strip"></div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["The board", "Bulk upload", "Insights", "About"])

# =================================================================
# TAB 1: PREDICT
# =================================================================
with tab1:
    st.subheader("Enter an employee's details")
    st.caption("Adjust the fields below, then check their risk score and what's behind it.")

    col_a, col_b, col_c = st.columns(3)
    input_data = {}

    with col_a:
        st.markdown("**Demographics**")
        input_data["Gender"] = st.selectbox("Gender", cat_options["Gender"])
        input_data["MaritalStatus"] = st.selectbox("Marital Status", cat_options["MaritalStatus"])
        input_data["Age"] = st.slider("Age", 18, 60, int(num_ranges["Age"]["default"]))
        input_data["DistanceFromHome"] = st.slider(
            "Distance From Home (miles)", 1, 30, int(num_ranges["DistanceFromHome"]["default"]))
        input_data["Education"] = st.slider("Education Level (1=Below College, 5=Doctor)", 1, 5,
                                             int(num_ranges["Education"]["default"]))
        input_data["EducationField"] = st.selectbox("Education Field", cat_options["EducationField"])

    with col_b:
        st.markdown("**Role & Tenure**")
        input_data["Department"] = st.selectbox("Department", cat_options["Department"])
        input_data["JobRole"] = st.selectbox("Job Role", cat_options["JobRole"])
        input_data["JobLevel"] = st.slider("Job Level", 1, 5, int(num_ranges["JobLevel"]["default"]))
        input_data["YearsAtCompany"] = st.slider(
            "Years At Company", 0, 40, int(num_ranges["YearsAtCompany"]["default"]))
        input_data["YearsInCurrentRole"] = st.slider(
            "Years In Current Role", 0, 20, int(num_ranges["YearsInCurrentRole"]["default"]))
        input_data["YearsSinceLastPromotion"] = st.slider(
            "Years Since Last Promotion", 0, 15, int(num_ranges["YearsSinceLastPromotion"]["default"]))
        input_data["YearsWithCurrManager"] = st.slider(
            "Years With Current Manager", 0, 20, int(num_ranges["YearsWithCurrManager"]["default"]))
        input_data["ManagerChangesLast2Yrs"] = st.slider(
            "Manager Changes (Last 2 Yrs)", 0, 5, int(num_ranges["ManagerChangesLast2Yrs"]["default"]))
        input_data["NumCompaniesWorked"] = st.slider(
            "Num Companies Worked Previously", 0, 9, int(num_ranges["NumCompaniesWorked"]["default"]))
        input_data["TotalWorkingYears"] = st.slider(
            "Total Working Years", 0, 40, int(num_ranges["TotalWorkingYears"]["default"]))

    with col_c:
        st.markdown("**Work & Satisfaction**")
        input_data["OverTime"] = st.selectbox("Works Overtime", cat_options["OverTime"])
        input_data["BusinessTravel"] = st.selectbox("Business Travel", cat_options["BusinessTravel"])
        input_data["MonthlyIncome"] = st.slider(
            "Monthly Income ($)", int(num_ranges["MonthlyIncome"]["min"]),
            int(num_ranges["MonthlyIncome"]["max"]), int(num_ranges["MonthlyIncome"]["default"]))
        input_data["PercentSalaryHike"] = st.slider(
            "Percent Salary Hike (last review)", 11, 25, int(num_ranges["PercentSalaryHike"]["default"]))
        input_data["StockOptionLevel"] = st.slider("Stock Option Level", 0, 3,
                                                     int(num_ranges["StockOptionLevel"]["default"]))
        input_data["EnvironmentSatisfaction"] = st.slider("Environment Satisfaction (1-4)", 1, 4,
                                                            int(num_ranges["EnvironmentSatisfaction"]["default"]))
        input_data["JobSatisfaction"] = st.slider("Job Satisfaction (1-4)", 1, 4,
                                                    int(num_ranges["JobSatisfaction"]["default"]))
        input_data["RelationshipSatisfaction"] = st.slider("Relationship Satisfaction (1-4)", 1, 4,
                                                             int(num_ranges["RelationshipSatisfaction"]["default"]))
        input_data["WorkLifeBalance"] = st.slider("Work Life Balance (1-4)", 1, 4,
                                                    int(num_ranges["WorkLifeBalance"]["default"]))
        input_data["JobInvolvement"] = st.slider("Job Involvement (1-4)", 1, 4,
                                                   int(num_ranges["JobInvolvement"]["default"]))
        input_data["PerformanceRating"] = st.slider("Performance Rating (3-4)", 3, 4,
                                                      int(num_ranges["PerformanceRating"]["default"]))
        input_data["TrainingTimesLastYear"] = st.slider("Training Times Last Year", 0, 6,
                                                          int(num_ranges["TrainingTimesLastYear"]["default"]))

    predict_btn = st.button("Make your move", type="primary", use_container_width=True)

    if predict_btn:
        # Build a single-row dataframe in the right column order, encode categoricals
        row = {}
        for c in cat_cols:
            le = encoders[c]
            row[c] = le.transform([str(input_data[c])])[0]
        for c in num_cols:
            row[c] = input_data[c]
        X_input = pd.DataFrame([row])[feature_cols]

        risk_proba = model.predict_proba(X_input)[0, 1]

        if risk_proba >= 0.5:
            level, glyph, verdict = "mate", "♚", "Checkmate risk"
        elif risk_proba >= 0.3:
            level, glyph, verdict = "check", "♞", "In check"
        else:
            level, glyph, verdict = "solid", "♙", "Solid position"

        st.divider()
        rcol1, rcol2 = st.columns([1, 2])

        with rcol1:
            st.markdown(f"""
            <div class="risk-card {level}">
              <div class="risk-glyph">{glyph}</div>
              <div class="risk-score">{risk_proba:.0%}</div>
              <div class="risk-verdict {level}">{verdict}</div>
            </div>
            """, unsafe_allow_html=True)

        with rcol2:
            st.markdown("**What's driving this score**")
            shap_vals = explainer.shap_values(X_input)
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            elif shap_vals.ndim == 3:
                sv = shap_vals[0, :, 1]
            else:
                sv = shap_vals[0]

            shap_df = pd.DataFrame({
                "Feature": feature_cols,
                "Value": [input_data[c] for c in feature_cols],
                "Impact": sv,
            }).sort_values("Impact", key=abs, ascending=False).head(8)

            rows_html = ""
            for _, r in shap_df.iterrows():
                cls = "shap-up" if r["Impact"] > 0 else "shap-down"
                arrow = "raises risk" if r["Impact"] > 0 else "lowers risk"
                rows_html += (f'<div class="shap-row"><span class="{cls}">{arrow}</span> '
                              f'— <b>{r["Feature"]}</b> = {r["Value"]} '
                              f'(impact {r["Impact"]:+.3f})</div>')
            st.markdown(rows_html, unsafe_allow_html=True)

# =================================================================
# TAB 2: INSIGHTS
# =================================================================
with tab2:
    st.subheader("Upload a roster")
    st.caption("Upload a CSV of employees to get a risk score and top drivers for "
               "everyone at once — useful for scanning a whole team or department.")

    with st.expander("Need the right format? Download a template"):
        template_cols = cat_cols + num_cols
        template_row = {c: cat_options[c][0] for c in cat_cols}
        template_row.update({c: num_ranges[c]["default"] for c in num_cols})
        template_df = pd.DataFrame([template_row])[template_cols]
        st.dataframe(template_df, use_container_width=True)
        st.download_button(
            "Download template CSV", template_df.to_csv(index=False),
            file_name="hrhelp_upload_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload employee CSV", type=["csv"])

    if uploaded is not None:
        try:
            up_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read that file: {e}")
            up_df = None

        if up_df is not None:
            missing = [c for c in feature_cols if c not in up_df.columns]
            if missing:
                st.error("This file is missing required columns: " + ", ".join(missing) +
                          ". Download the template above to see the exact column names needed.")
            else:
                work_df = up_df.copy()
                unseen_report = {}

                # Safe-encode categoricals: unseen category values fall back to a
                # known default rather than crashing, and get flagged for the user
                for c in cat_cols:
                    le = encoders[c]
                    known = set(le.classes_)
                    vals = work_df[c].astype(str)
                    unseen_vals = vals[~vals.isin(known)].unique().tolist()
                    if unseen_vals:
                        unseen_report[c] = unseen_vals
                    fallback = le.classes_[0]
                    cleaned = vals.apply(lambda v: v if v in known else fallback)
                    work_df[c] = le.transform(cleaned)

                # Numeric columns: coerce, fill any bad/missing values with the
                # training median rather than dropping the row
                for c in num_cols:
                    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")
                    if work_df[c].isna().any():
                        work_df[c] = work_df[c].fillna(num_ranges[c]["default"])

                X_batch = work_df[feature_cols]

                if unseen_report:
                    msg = "; ".join(f"{k}: {', '.join(v)}" for k, v in unseen_report.items())
                    st.warning(f"Some values weren't in the training data and were treated "
                               f"as the nearest known category — {msg}")

                probs = model.predict_proba(X_batch)[:, 1]
                shap_vals_batch = explainer.shap_values(X_batch)
                if isinstance(shap_vals_batch, list):
                    sv_batch = shap_vals_batch[1]
                elif shap_vals_batch.ndim == 3:
                    sv_batch = shap_vals_batch[:, :, 1]
                else:
                    sv_batch = shap_vals_batch

                results = up_df.copy()
                results["Risk score"] = (probs * 100).round(1)
                results["Verdict"] = pd.cut(
                    probs, bins=[-0.01, 0.3, 0.5, 1.01],
                    labels=["Solid position", "In check", "Checkmate risk"])

                top_drivers = []
                for i in range(len(X_batch)):
                    row_sv = sv_batch[i]
                    top_idx = np.argsort(np.abs(row_sv))[::-1][:2]
                    parts = []
                    for idx in top_idx:
                        direction = "raises" if row_sv[idx] > 0 else "lowers"
                        parts.append(f"{feature_cols[idx]} ({direction} risk)")
                    top_drivers.append("; ".join(parts))
                results["Top drivers"] = top_drivers

                results = results.sort_values("Risk score", ascending=False)

                st.divider()
                n_high = (results["Verdict"] == "Checkmate risk").sum()
                n_mod = (results["Verdict"] == "In check").sum()
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Employees scanned", len(results))
                mcol2.metric("Checkmate risk", int(n_high))
                mcol3.metric("In check", int(n_mod))

                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download results CSV", results.to_csv(index=False),
                    file_name="hrhelp_bulk_risk_results.csv", mime="text/csv",
                    type="primary")

# =================================================================
# TAB 3: INSIGHTS
# =================================================================
with tab3:
    st.subheader("Model performance")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown("**Random forest (deployed model)**")
        st.json(metrics["random_forest"])
    with mcol2:
        st.markdown("**Logistic regression (baseline)**")
        st.json(metrics["logistic_regression"])

    st.divider()
    st.subheader("Attrition rate by segment")
    eda = load_eda_summary()
    seg_choice = st.selectbox("Choose a segment to explore", eda["segment"].unique())
    seg_data = eda[eda["segment"] == seg_choice].sort_values("attrition_rate", ascending=False)
    st.bar_chart(seg_data.set_index("value")["attrition_rate"])
    st.dataframe(seg_data[["value", "attrition_rate", "n"]], use_container_width=True)

    st.divider()
    st.subheader("Findings and recommendations")
    st.markdown(load_insights_report())

# =================================================================
# TAB 4: ABOUT
# =================================================================
with tab4:
    st.subheader("About HRHELP")
    st.markdown("""
HRHELP reads the board before a piece is lost: it estimates the odds that a
given employee will leave, and explains which factors are pushing that
estimate up or down — so HR can move before someone hands in notice, not after.

**How it works**
1. Simulates three messy real-world HR data sources (HRIS, payroll, tenure/manager
   systems) with mismatched ID formats, missing values, duplicates, and inconsistent text
2. Cleans and merges them into one analysis-ready dataset, with every fix logged
3. Runs statistical testing (chi-square, logistic regression) to identify
   significant attrition drivers
4. Turns the statistical results into a written findings and recommendations report
5. Trains a random forest classifier, evaluated on precision, recall, and ROC-AUC —
   not accuracy alone, since attrition is a ~20% minority class
6. Adds SHAP explainability so every prediction comes with a reason, not just a number

**Built with:** Python, pandas, scikit-learn, statsmodels, SHAP, Streamlit

**Worth knowing**
- The dataset is synthetic, generated to carry the same statistical relationships
  as the well-known IBM HR Attrition dataset
- These are associations, not proven causes
- Any action touching a protected characteristic should go through legal and
  ethical review before real-world use
""")
