import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

# Load model
model = joblib.load("credit_risk_model.pkl")

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

st.title("💳 Credit Risk Prediction System")
st.write("Predict whether a loan applicant is **High Risk** or **Low Risk**")

st.divider()

# Input form
with st.form("prediction_form"):

    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)

    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Term (months)", min_value=0)

    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict Risk")

if submitted:

    Total_Income = ApplicantIncome + CoapplicantIncome
    Debt_to_Income = LoanAmount / Total_Income if Total_Income > 0 else 0

    input_data = pd.DataFrame([{
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
        "Total_Income": Total_Income,
        "Debt_to_Income": Debt_to_Income,
        "Age_Group": "Adult"   # same dummy feature used in training
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ High Risk Applicant\n\nRisk Probability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk Applicant\n\nRisk Probability: {probability:.2f}")