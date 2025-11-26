import streamlit as st
import pickle
import numpy as np
import torch
import json

from utils import preprocess_input   # bizim preprocessing funksiyamız


# -----------------------------------------------------------
# LOAD TRAINED (CALIBRATED) MODEL + ENCODERS + SCALER
# -----------------------------------------------------------
model = pickle.load(open("xgb_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

FEATURES = [
    "Tumor Size",
    "Regional Node Positive",
    "T Stage",
    "N Stage",
    "differentiate",
    "Grade",
    "Estrogen Status",
    "Progesterone Status",
    "Race"
]

# -----------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Survival Prediction",
                   page_icon="🩺",
                   layout="centered")


# -----------------------------------------------------------
# HEADER (Medical Pastel Theme)
# -----------------------------------------------------------
st.markdown("""
    <div style="
        background-color:#DFF5E3;
        padding:18px;
        border-radius:10px;
        text-align:center;
        border: 1px solid #B7E4C7;
        margin-bottom: 15px;
    ">
        <h1 style="color:#0C513F; margin:0; font-size:27px;">
            🩺 Breast Cancer Survival Prediction (Calibrated XGBoost)
        </h1>
    </div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# DESCRIPTION BOX
# -----------------------------------------------------------
st.markdown("""
<div style="
    background-color:#F2FBF5;
    padding:15px;
    border-radius:10px;
    border-left:4px solid #66C2A5;
    font-size:16px;
">
Bu sistem döş xərçəngi xəstələri üçün **1-year survival probability** (sağ qalma ehtimalı)
hesablayır. Model **Platt calibration** ilə kalibrasiya edilib, bu da ehtimalların daha
stabil və tibbi real olmasını təmin edir.

Model aşağıdakı risk bölgüsündən istifadə edir:

🟢 <b>Aşağı Risk:</b> P(survival) ≥ 0.80  
🟡 <b>Orta Risk:</b> 0.50 ≤ P(survival) < 0.80  
🔴 <b>Yüksək Risk:</b> P(survival) < 0.50  

Model yalnız 9 ən vacib klinik göstəricidən istifadə edir.
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# -----------------------------------------------------------
# USER INPUT FORM
# -----------------------------------------------------------
st.subheader("📥 Dəyərləri daxil edin")

user_input = {}
col1, col2 = st.columns(2)

# NUMERICAL
with col1:
    user_input["Tumor Size"] = st.number_input("Tumor Size (mm)", 1, 200, 20)

with col2:
    user_input["Regional Node Positive"] = st.number_input("Positive Lymph Nodes", 0, 30, 0)

# CATEGORICAL
with col1:
    user_input["T Stage"] = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])

with col2:
    user_input["N Stage"] = st.selectbox("N Stage", ["N1", "N2", "N3"])

with col1:
    user_input["differentiate"] = st.selectbox(
        "Differentiate",
        ["Poorly differentiated", "Moderately differentiated", "Well differentiated", "Undifferentiated"]
    )

with col2:
    user_input["Grade"] = st.selectbox("Grade", ["1", "2", "3", " anaplastic; Grade IV"])

with col1:
    user_input["Estrogen Status"] = st.selectbox("Estrogen Status", ["Positive", "Negative"])

with col2:
    user_input["Progesterone Status"] = st.selectbox("Progesterone Status", ["Positive", "Negative"])

with col1:
    user_input["Race"] = st.selectbox("Race", ["White", "Black", "Other"])

st.markdown("---")


# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------
if st.button("🔮 Proqnoz et"):

    X = preprocess_input(user_input, encoders, scaler)

    prob = model.predict_proba(X)[0][1]     # calibrated survival probability

    # RISK STRATIFICATION
    if prob >= 0.80:
        st.success(f"🟢 Aşağı Risk — **{prob:.2f}** (Yüksək sağ qalma ehtimalı)")
    elif prob >= 0.50:
        st.warning(f"🟡 Orta Risk — **{prob:.2f}** (Orta sağ qalma ehtimalı)")
    else:
        st.error(f"🔴 Yüksək Risk — **{prob:.2f}** (Aşağı sağ qalma ehtimalı)")

    st.write("---")

    st.subheader("🧪 Model Input Vector (DEBUG)")
    st.write(X)


# -----------------------------------------------------------
# ALWAYS-VISIBLE RESULTS (ACCORDION)
# -----------------------------------------------------------
st.markdown("---")

with st.expander("📊 Confusion Matrix"):
    st.image("images/confusion_matrix.png", width=520)

with st.expander("📈 ROC Curve"):
    st.image("images/roc_curve.png", width=520)

with st.expander("📉 Calibration Curve"):
    st.image("images/calibration_curve.png", width=520)

with st.expander("🔥 Feature Importance (XGBoost)"):
    st.image("images/xgb_feature_importance_top10.png", width=520)

with st.expander("🧠 SHAP Summary Plot"):
    st.image("images/xgb_shap_summary.png", width=520)


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.caption("Developed by Etibar Vazirov — Calibrated ML · Clinical AI · 2025")
