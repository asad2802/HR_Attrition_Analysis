# HR Attrition Analysis & Prediction
https://hrmetricsdata.streamlit.app/

**Business problem:** Identify which factors actually predict employee attrition so HR can prioritize retention spend where it matters most.

## Project structure
- `data/raw/` — 3 messy source tables simulating separate HR systems (HRIS, Payroll, Tenure/Manager)
- `data/processed/` — cleaned, merged master dataset
- `src/` — data generation, cleaning, merging, modeling scripts
- `app/` — Streamlit web app (interactive attrition risk predictor + SHAP explanations)

## Setup
```
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline
```
python src/generate_data.py
python src/make_messy_sources.py
python src/clean_and_merge.py      # (next step)
python src/train_model.py          # (next step)
```

## Run the app locally
```
streamlit run app/app.py
```

## Status
- [x] Data generation (synthetic, IBM-dataset-realistic)
- [x] Messy multi-source simulation
- [ ] Clean & merge pipeline
- [ ] EDA + statistical testing
- [ ] Predictive model + SHAP
- [ ] Streamlit app
- [ ] Deployment
