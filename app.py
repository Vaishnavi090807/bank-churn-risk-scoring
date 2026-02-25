import json
import numpy as np
import pandas as pd
import streamlit as st

from src.utils import load_pickle, load_json
from src.helpers import add_engineered_features, risk_score, risk_label


# -----------------------------
# CONFIG (Exact paths)
# -----------------------------
ARTIFACT_DIR = "artifacts"
REPORTS_DIR = "reports"

BEST_MODEL_PATH = f"{ARTIFACT_DIR}/best_model.pkl"
METRICS_PATH = f"{ARTIFACT_DIR}/metrics.json"

# optional reports (if available)
NATIVE_IMPORTANCE_CSV = f"{REPORTS_DIR}/feature_importance_native.csv"
PERM_IMPORTANCE_CSV = f"{REPORTS_DIR}/feature_importance_permutation.csv"


def load_threshold(metrics_path: str) -> float:
    """
    Read tuned threshold from artifacts/metrics.json (Step 3 output).
    Fallback to 0.50 if not found.
    """
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        t = m.get("BEST_MODEL", {}).get("tuned_threshold_result", {}).get("threshold", 0.50)
        return float(t)
    except Exception:
        return 0.50


def safe_read_csv(path: str):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


st.set_page_config(page_title="Bank Churn Risk Scoring", layout="wide")

st.title("🏦 Bank Customer Churn — Predictive Modeling & Risk Scoring")

# Load model + threshold
pipe = load_pickle(BEST_MODEL_PATH)
threshold = load_threshold(METRICS_PATH)

st.caption(f"Using tuned threshold for HIGH risk = **{threshold:.2f}** (from artifacts/metrics.json)")

# Sidebar inputs
st.sidebar.header("Customer Inputs")

credit_score = st.sidebar.number_input("CreditScore", min_value=300, max_value=900, value=650, step=1)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35, step=1)
tenure = st.sidebar.number_input("Tenure (years)", min_value=0, max_value=10, value=3, step=1)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
num_products = st.sidebar.number_input("NumOfProducts", min_value=1, max_value=4, value=2, step=1)
has_cr_card = st.sidebar.selectbox("HasCrCard", [0, 1], index=1)
is_active = st.sidebar.selectbox("IsActiveMember", [0, 1], index=1)
salary = st.sidebar.number_input("EstimatedSalary", min_value=0.0, value=70000.0, step=1000.0)

st.sidebar.markdown("---")
what_if = st.sidebar.checkbox("Enable What-If Simulator", value=True)

# Build single-row input dataframe (raw)
input_df = pd.DataFrame([{
    "Year": tenure,                 # ✅ ADD THIS (same value)
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,               # ✅ keep this also
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary
}])

# Add engineered features (must match training)
input_df = add_engineered_features(input_df)

# Predict
proba = float(pipe.predict_proba(input_df)[0, 1])
score = risk_score(proba)
label = risk_label(proba, threshold)

# Layout
col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Prediction")
    st.metric("Churn Probability", f"{proba:.3f}")
    st.metric("Risk Score (0–100)", f"{score}")
    st.metric("Risk Label", label)

    st.write("**Interpretation:**")
    st.write(
        "- Probability is model output (0 to 1).\n"
        "- Risk Score is probability × 100.\n"
        f"- HIGH risk if probability ≥ **{threshold:.2f}** (tuned to reduce false positives)."
    )

with col2:
    st.subheader("Probability Gauge")
    st.progress(min(max(proba, 0.0), 1.0))
    st.write("Low → Medium → High as probability increases.")

st.divider()

# What-if simulator
if what_if:
    st.subheader("What-If Scenario Simulator")
    st.write("Change a few factors and see how churn probability shifts.")

    wcol1, wcol2, wcol3 = st.columns(3)

    with wcol1:
        delta_age = st.slider("Adjust Age (+/-)", -10, 10, 0, 1)
        delta_credit = st.slider("Adjust CreditScore (+/-)", -150, 150, 0, 10)

    with wcol2:
        delta_balance = st.slider("Adjust Balance (+/-)", -100000, 100000, 0, 5000)
        delta_salary = st.slider("Adjust EstimatedSalary (+/-)", -100000, 100000, 0, 5000)

    with wcol3:
        new_active = st.selectbox("Set IsActiveMember to", [0, 1], index=is_active)
        new_products = st.slider("Set NumOfProducts to", 1, 4, int(num_products), 1)

    scenario = pd.DataFrame([{
        "Year": int(tenure),            # ✅ ADD THIS
        "CreditScore": int(credit_score + delta_credit),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age + delta_age),
        "Tenure": int(tenure),          # ✅ keep this also
        "Balance": float(balance + delta_balance),
        "NumOfProducts": int(new_products),
        "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(new_active),
        "EstimatedSalary": float(salary + delta_salary),
    }])

    scenario = add_engineered_features(scenario)
    scenario_proba = float(pipe.predict_proba(scenario)[0, 1])

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Original Probability", f"{proba:.3f}")
    with s2:
        st.metric("Scenario Probability", f"{scenario_proba:.3f}")
    with s3:
        st.metric("Change (Δ)", f"{(scenario_proba - proba):+.3f}")

st.divider()

# Explainability panel from saved CSVs
st.subheader("Explainability (Top Drivers)")
native_df = safe_read_csv(NATIVE_IMPORTANCE_CSV)
perm_df = safe_read_csv(PERM_IMPORTANCE_CSV)

tab1, tab2 = st.tabs(["Native Importance (encoded)", "Permutation Importance (raw)"])

with tab1:
    if native_df is None:
        st.info("Native importance CSV not found. Run Step 4 to generate reports/feature_importance_native.csv")
    else:
        st.write("Top 15 features (after encoding & scaling).")
        st.dataframe(native_df.sort_values("importance", ascending=False).head(15), use_container_width=True)

with tab2:
    if perm_df is None:
        st.info("Permutation importance CSV not found. Run Step 4 to generate reports/feature_importance_permutation.csv")
    else:
        st.write("Top 15 raw features (business-friendly).")
        st.dataframe(perm_df.sort_values("perm_importance_mean", ascending=False).head(15), use_container_width=True)

st.caption("Note: Permutation importance is usually easier to explain to faculty since it uses original columns.")