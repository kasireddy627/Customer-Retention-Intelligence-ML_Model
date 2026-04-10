import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# -------------------- HEADER --------------------
st.markdown("""
<style>
.main-header {
    background-color: #1f4e79;
    padding: 15px;
    border-radius: 8px;
}
.main-header h1 {
    color: white;
    margin: 0;
    font-size: 26px;
}
.desc {
    margin-top: 10px;
    font-size: 15px;
    color: #ddd;
}
</style>

<div class="main-header">
    <h1>Customer Churn Prediction System</h1>
</div>

<div class="desc">
Predicts customers likely to leave using behavior and billing patterns.<br>
Helps businesses reduce revenue loss through early retention actions.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

artifact = load_model()
model = artifact["model"]
threshold = artifact["threshold"]

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("Customer Inputs")

gender = st.sidebar.selectbox("Gender ?", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen ?", [0, 1], format_func=lambda x: "Yes" if x else "No")
Partner = st.sidebar.selectbox("Partner ?", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents ?", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months) ?", 0, 72, 12)

st.sidebar.divider()

PhoneService = st.sidebar.selectbox("Phone Service ?", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines ?", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service ?", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.sidebar.selectbox("Online Security ?", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup ?", ["Yes", "No", "No internet service"])

st.sidebar.divider()

DeviceProtection = st.sidebar.selectbox("Device Protection ?", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support ?", ["Yes", "No", "No internet service"])

StreamingTV = st.sidebar.selectbox("Streaming TV ?", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies ?", ["Yes", "No", "No internet service"])

Contract = st.sidebar.selectbox("Contract ?", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing ?", ["Yes", "No"])

PaymentMethod = st.sidebar.selectbox(
    "Payment Method ?",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges ?", 0.0, 200.0, 50.0)

# -------------------- INPUT BUILDER --------------------
def build_input():
    df = pd.DataFrame([{
        "gender": str(gender),
        "SeniorCitizen": str(SeniorCitizen),
        "Partner": str(Partner),
        "Dependents": str(Dependents),
        "tenure": float(tenure),
        "PhoneService": str(PhoneService),
        "MultipleLines": str(MultipleLines),
        "InternetService": str(InternetService),
        "OnlineSecurity": str(OnlineSecurity),
        "OnlineBackup": str(OnlineBackup),
        "DeviceProtection": str(DeviceProtection),
        "TechSupport": str(TechSupport),
        "StreamingTV": str(StreamingTV),
        "StreamingMovies": str(StreamingMovies),
        "Contract": str(Contract),
        "PaperlessBilling": str(PaperlessBilling),
        "PaymentMethod": str(PaymentMethod),
        "MonthlyCharges": float(MonthlyCharges)
    }])

    df.replace(["", "None", "nan"], np.nan, inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(0)

    return df

# -------------------- PREDICTION --------------------
if st.button("Predict Churn"):

    input_df = build_input()
    prob = model.predict_proba(input_df)[:, 1][0]

    st.subheader("Prediction Result")

    if prob >= 0.7:
        st.error(f"High Risk | Probability: {prob:.2f}")
    elif prob >= threshold:
        st.warning(f"Medium Risk | Probability: {prob:.2f}")
    else:
        st.success(f"Low Risk | Probability: {prob:.2f}")

st.markdown("---")

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["What & Why", "Model Metrics", "Threshold Logic"])

# -------- TAB 1 --------
with tab1:
    st.subheader("What is this system?")
    st.write("""
This system predicts whether a customer is likely to churn (leave the service) based on their usage, services, and billing behavior.
""")

    st.subheader("What problem does it solve?")
    st.write("""
Businesses lose revenue when customers leave without warning. This model helps identify at-risk customers early so action can be taken before churn happens.
""")

    st.subheader("Why this approach?")
    st.write("""
- Early detection enables targeted retention strategies  
- Reduces revenue loss  
- Focuses efforts only on high-risk customers  
""")

# -------- TAB 2 --------
with tab2:
    st.subheader("Model Performance Comparison")

    df_metrics = pd.DataFrame({
        "Model": ["XGB", "LogReg", "SVC", "GB", "KNN", "RF"],
        "Accuracy": [0.73, 0.72, 0.72, 0.79, 0.76, 0.78],
        "Precision": [0.49, 0.49, 0.48, 0.63, 0.54, 0.61],
        "Recall": [0.78, 0.78, 0.78, 0.52, 0.57, 0.49],
        "F1 Score": [0.60, 0.60, 0.60, 0.57, 0.56, 0.54]
    })

    st.dataframe(df_metrics)

    st.write("""
XGBoost was selected because it provides the best balance between recall and overall performance.
""")

# -------- TAB 3 --------
with tab3:
    st.subheader("Threshold Strategy (0.5 vs 0.4)")

    st.write("""
Lowering threshold from 0.5 to 0.4 increases churn detection but also increases false positives.
""")

    st.table(pd.DataFrame({
        "Metric": ["Recall", "Precision", "Accuracy"],
        "0.5": [0.79, 0.50, 0.73],
        "0.4": [0.84, 0.46, 0.70]
    }))

    st.write("""
Business Impact:
- 0.4 threshold catches more churn customers  
- Slight drop in precision is acceptable to prevent revenue loss  
""")