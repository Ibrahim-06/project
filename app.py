import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("Logistic_model.pkl")
scaler = joblib.load("Scaler.pkl")
feature_cols = joblib.load("features.pkl")

st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}

@keyframes floatText {
  0%   {transform: translateY(0);}
  50%  {transform: translateY(-10px);}
  100% {transform: translateY(0);}
}

h1 {
    animation: floatText 3s ease-in-out infinite;
    text-align: center;
    color: white;
    font-size: 3em;
}

.stSelectbox, .stSlider, .stTextInput {
    background-color: #2f927a !important;
    border-radius: 10px !important;
    padding: 15px !important;
    color: black !important;
    transition: 0.2s;
    margin-bottom:7px;
}
.stSelectbox:hover, .stSlider:hover, .stTextInput:hover {
    background-color: #c9ad36 !important;
}

.stSlider > div[data-baseweb="slider"] > div {
    # background: linear-gradient(to right, #00c6ff, #b9646e);
    border-radius: 100px;
    height: 2rem;
}
.stSlider > div[data-baseweb="slider"] span {
    background-color: green !important;
    # border: 2px solid white;
}

.stButton>button {
    background-color: #c9ad36;
    color: white;
    border-radius: 12px;
    padding: 0.5em 2em;
    font-weight: bold;
    font-size: 1.1em;
    transition: 0.3s;
    margin-left:45%;
}
.stButton>button:hover {
    background-color: #2f927a;
    color: black;
    transform: scale(1.1);
}

.output-box {
    background-color: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 15px;
    margin-top: 30px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# ✨ Customer Churn Prediction App ✨")

st.markdown("### 🧾 Customer Information")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col2:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

with col3:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.markdown("### 💵 Billing Details")
colA, colB, colC = st.columns(3)
with colA:
    tenure = st.slider("📅 Tenure (Months)", 0, 72, 12)
with colB:
    MonthlyCharges = st.slider("💰 Monthly Charges", 0.0, 120.0, 70.0)
with colC:
    TotalCharges = st.slider("💵 Total Charges", 0.0, 10000.0, 350.0)

data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

input_df = pd.DataFrame([data])
input_df_encoded = pd.get_dummies(input_df)

for col in feature_cols:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0
input_df_encoded = input_df_encoded[feature_cols]

if st.button("🔍 Predict Churn"):
    input_scaled = scaler.transform(input_df_encoded)
    proba = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if proba >= 0.28 else 0

    st.markdown('<div class="output-box">', unsafe_allow_html=True)
    st.markdown("### 🔎 Result:")
    if prediction == 1:
        st.error("🔴 The customer is **likely to churn**.")
    else:
        st.success("🟢 The customer is **not likely to churn**.")

    st.markdown(f"📈 **Churn probability: {proba * 100:.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)
