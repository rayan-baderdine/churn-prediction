import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load model and features
with open('model/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")
st.title("📊 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict churn probability.")

# --- Input form ---
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

# --- Build input row ---
def build_input():
    row = {f: 0 for f in feature_names}

    row['tenure'] = tenure
    row['MonthlyCharges'] = monthly_charges
    row['TotalCharges'] = total_charges
    row['SeniorCitizen'] = senior_citizen
    row['Partner'] = 1 if partner == "Yes" else 0
    row['Dependents'] = 1 if dependents == "Yes" else 0
    row['PhoneService'] = 1 if phone_service == "Yes" else 0
    row['PaperlessBilling'] = 1 if paperless_billing == "Yes" else 0
    row['OnlineSecurity'] = 1 if online_security == "Yes" else 0
    row['TechSupport'] = 1 if tech_support == "Yes" else 0
    row['MultipleLines'] = 1 if multiple_lines == "Yes" else 0

    # Internet service one-hot
    row['InternetService_DSL'] = 1 if internet_service == "DSL" else 0
    row['InternetService_Fiber optic'] = 1 if internet_service == "Fiber optic" else 0
    row['InternetService_No'] = 1 if internet_service == "No" else 0

    # Contract one-hot
    row['Contract_Month-to-month'] = 1 if contract == "Month-to-month" else 0
    row['Contract_One year'] = 1 if contract == "One year" else 0
    row['Contract_Two year'] = 1 if contract == "Two year" else 0

    # Payment method one-hot
    row['PaymentMethod_Electronic check'] = 1 if payment_method == "Electronic check" else 0
    row['PaymentMethod_Mailed check'] = 1 if payment_method == "Mailed check" else 0
    row['PaymentMethod_Bank transfer (automatic)'] = 1 if payment_method == "Bank transfer (automatic)" else 0
    row['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == "Credit card (automatic)" else 0

    return pd.DataFrame([row])[feature_names]

# --- Predict button ---
if st.button("🔍 Predict Churn"):
    input_df = build_input()
    proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ High churn risk — {proba*100:.1f}% probability")
    else:
        st.success(f"✅ Low churn risk — {proba*100:.1f}% probability")

    # SHAP explanation
    st.subheader("Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()