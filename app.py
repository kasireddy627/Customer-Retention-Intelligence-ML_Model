import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors


# App configuration
st.set_page_config(
    page_title="Churn Prediction & Retention Analytics Platform",
    layout="wide"
)


# UI header (clean, no logo, no company line)
st.markdown("## Churn Prediction & Retention Analytics Platform")


st.info(
    "How to use this platform:\n\n"
    "1. Enter customer details in the left panel\n"
    "2. Review churn probability and prediction results\n"
    "3. Download the customer retention report\n"
    "4. Review the key factors influencing churn below"
)


# Load trained model
@st.cache_resource
def load_model():
    with open("churn_xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

artifact = load_model()
model = artifact["model"]
threshold = artifact["threshold"]


# Sidebar – customer inputs
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"],
    help="Customer gender as recorded in the system."
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x else "No",
    help="Select Yes if the customer is aged 60 years or above."
)

partner = st.sidebar.selectbox(
    "Has Partner",
    ["Yes", "No"],
    help="Whether the customer has a spouse or partner."
)

dependents = st.sidebar.selectbox(
    "Has Dependents",
    ["Yes", "No"],
    help="Whether the customer supports dependents."
)

tenure = st.sidebar.slider(
    "Customer Tenure (Months)",
    0, 72, 12,
    help="How long the customer has been with the company."
)

st.sidebar.divider()

phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.sidebar.divider()

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
    help="Short-term contracts generally have higher churn risk."
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"],
    help="Paperless billing customers historically churn more."
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ],
    help="Electronic check payments are associated with higher churn."
)

monthly = st.sidebar.slider(
    "Monthly Charges",
    20.0, 120.0, 70.0,
    help="Average monthly bill amount."
)

total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=monthly * max(tenure, 1),
    help="Total amount charged to the customer so far."
)


# Build input dataframe
input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "tenure_bucket": pd.cut(
        [tenure],
        [0, 6, 12, 24, 100],
        labels=["0-6", "6-12", "12-24", "24+"]
    )[0],
    "avg_monthly_spend": total / max(tenure, 1),
    "is_month_to_month": 1 if contract == "Month-to-month" else 0,
    "is_electronic_check": 1 if payment == "Electronic check" else 0,
    "num_services": sum([
        phone == "Yes",
        multiple == "Yes",
        internet != "No",
        online_security == "Yes",
        online_backup == "Yes",
        device_protection == "Yes",
        tech_support == "Yes",
        streaming_tv == "Yes",
        streaming_movies == "Yes"
    ])
}])


# Prediction
proba = model.predict_proba(input_df)[0, 1]
prediction = int(proba >= threshold)

churn_label = "YES" if prediction else "NO"
risk = "High Risk" if prediction else "Low Risk"


# Prediction results
st.subheader("Prediction Results")

with st.container():
    st.markdown(
        """
<div style="
    background-color: rgba(28, 131, 225, 0.08);
    padding: 25px;
    border-radius: 12px;
">

<h2>What do these results mean?</h2>

<p><strong>Churn Probability</strong><br>
The likelihood that this customer may leave the service in the near future.</p>

<p><strong>Decision Threshold</strong><br>
The cutoff value used by the business to decide whether a customer should be classified as churn risk.</p>

<p><strong>Churn Prediction</strong></p>
<ul>
<li><strong>YES</strong>: Customer is likely to churn</li>
<li><strong>NO</strong>: Customer is unlikely to churn</li>
</ul>

<p><strong>Risk Category</strong><br>
Indicates how urgently the customer should be targeted for retention actions.</p>

</div>
""",
        unsafe_allow_html=True
    )


c1, c2, c3, c4 = st.columns(4)
c1.metric("Churn Probability", f"{proba:.2%}")
c2.metric("Decision Threshold", f"{threshold:.2f}")
c3.metric("Churn Prediction", churn_label)
c4.metric("Risk Category", risk)


# SHAP computation
X_transformed = model.named_steps["prep"].transform(input_df)

ohe = model.named_steps["prep"].named_transformers_["cat"]
cat_features = ohe.get_feature_names_out(
    model.named_steps["prep"].transformers_[1][2]
)
num_features = model.named_steps["prep"].transformers_[0][2]
feature_names = np.concatenate([num_features, cat_features])

def model_predict(data):
    return model.named_steps["model"].predict_proba(data)[:, 1]

explainer = shap.Explainer(
    model_predict,
    X_transformed,
    feature_names=feature_names,
    max_evals=200
)

shap_values = explainer(X_transformed[:1])


# Business-readable explanations
FEATURE_EXPLANATIONS = {
    "Contract_Month-to-month": "Customer is on a month-to-month contract",
    "tenure": "Customer has low tenure",
    "MonthlyCharges": "Customer pays higher monthly charges",
    "PaymentMethod_Electronic check": "Customer uses electronic check payment",
    "TechSupport_No": "Customer does not have technical support",
    "OnlineSecurity_No": "Customer does not have online security"
}

shap_df = pd.DataFrame({
    "feature": feature_names,
    "impact": shap_values[0].values
})

shap_df["abs_impact"] = shap_df["impact"].abs()
top_shap = shap_df.sort_values("abs_impact", ascending=False).head(5)

def explain_feature(row):
    base = FEATURE_EXPLANATIONS.get(row["feature"], row["feature"].replace("_", " "))
    direction = "increases churn risk" if row["impact"] > 0 else "reduces churn risk"
    return f"{base} ({direction})"

top_shap["business_reason"] = top_shap.apply(explain_feature, axis=1)


# PDF generator (branded)
def generate_pdf():
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []

    header = Table(
        [[
            Paragraph("Customer Retention Intelligence Report", styles["Title"])
        ]],
        colWidths=[330, 130]
    )

    header.setStyle(TableStyle([
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    elements.append(header)
    elements.append(Spacer(1, 10))

    elements.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%d %b %Y, %H:%M')}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))

    summary = Table([
        ["Metric", "Value"],
        ["Churn Probability", f"{proba:.2%}"],
        ["Decision Threshold", f"{threshold:.2f}"],
        ["Churn Prediction", churn_label],
        ["Risk Category", risk]
    ], colWidths=[200, 260])

    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.8, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke)
    ]))

    content = [summary, Spacer(1, 15)]

    content.append(Paragraph("Key Factors Affecting Churn", styles["Heading2"]))
    content.append(Spacer(1, 8))

    for r in top_shap["business_reason"]:
        content.append(Paragraph(f"- {r}", styles["Normal"]))

    frame = Table([[content]], colWidths=[460])
    frame.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.5, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 15),
        ("RIGHTPADDING", (0, 0), (-1, -1), 15),
        ("TOPPADDING", (0, 0), (-1, -1), 15),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 15)
    ]))

    elements.append(frame)

    doc.build(elements)
    buffer.seek(0)
    return buffer


# Download PDF
st.subheader("Download Customer Report")

st.download_button(
    "Download PDF Report",
    data=generate_pdf(),
    file_name="customer_retention_intelligence_report.pdf",
    mime="application/pdf"
)


# SHAP visualization
st.subheader("Key Factors Affecting Churn")

st.caption(
    "Positive values increase churn risk. "
    "Negative values reduce churn risk."
)

fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
st.pyplot(fig)

st.caption(
    "This prediction supports decision-making and should be "
    "used alongside business judgment."
)
