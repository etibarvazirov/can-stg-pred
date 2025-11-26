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

user_input["Tumor Size"] = st.number_input(
    "Tumor Size (mm)", 
    1, 200, 
    key="tumor_size"
)

user_input["Reginol Node Positive"] = st.number_input(
    "Reginol Node Positive", 
    0, 30, 
    key="rnp"
)

user_input["T Stage"] = st.selectbox(
    "T Stage", 
    ["T1", "T2", "T3", "T4"],
    key="t_stage"
)

user_input["N Stage"] = st.selectbox(
    "N Stage", 
    ["N1", "N2", "N3"],
    key="n_stage"
)

user_input["differentiate"] = st.selectbox(
    "Differentiate",
    ["Poorly differentiated", "Moderately differentiated", 
     "Well differentiated", "Undifferentiated"],
    key="diff"
)

user_input["Grade"] = st.selectbox(
    "Grade",
    ["1", "2", "3", " anaplastic; Grade IV"],
    key="grade"
)

user_input["Estrogen Status"] = st.selectbox(
    "Estrogen Status", ["Positive", "Negative"],
    key="er"
)

user_input["Progesterone Status"] = st.selectbox(
    "Progesterone Status", ["Positive", "Negative"],
    key="pr"
)

user_input["Race"] = st.selectbox(
    "Race", ["White", "Black", "Other"],
    key="race"
)


# CLEAN KEYS → prevent KeyError
clean_input = {k.strip(): v for k, v in user_input.items()}
user_input = clean_input

st.markdown("---")

# Ready presets ----------------------------------------------
st.subheader("📌 Hazır nümunələr (Presets)")

colA, colB, colC = st.columns(3)

if colA.button("🟢 Low Risk"):
    st.session_state.tumor_size = 8
    st.session_state.rnp = 0
    st.session_state.t_stage = "T1"
    st.session_state.n_stage = "N1"
    st.session_state.diff = "Well differentiated"
    st.session_state.grade = "1"
    st.session_state.er = "Positive"
    st.session_state.pr = "Positive"
    st.session_state.race = "White"
    st.rerun()

if colB.button("🟡 Medium Risk"):
    st.session_state.tumor_size = 38
    st.session_state.rnp = 4
    st.session_state.t_stage = "T2"
    st.session_state.n_stage = "N2"
    st.session_state.diff = "Moderately differentiated"
    st.session_state.grade = "2"
    st.session_state.er = "Positive"
    st.session_state.pr = "Negative"
    st.session_state.race = "Other"
    st.rerun()

if colC.button("🔴 High Risk"):
    st.session_state.tumor_size = 90
    st.session_state.rnp = 12
    st.session_state.t_stage = "T3"
    st.session_state.n_stage = "N3"
    st.session_state.diff = "Poorly differentiated"
    st.session_state.grade = "3"
    st.session_state.er = "Negative"
    st.session_state.pr = "Negative"
    st.session_state.race = "Black"
    st.rerun()


# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------
if st.button("🔮 Proqnoz et"):

    X = preprocess_input(user_input, encoders, scaler)

    prob = model.predict_proba(X)[0][1]     # calibrated survival probability

    # RISK STRATIFICATION (3-level)
    if prob >= 0.92:
        st.success(f"🟢 Aşağı Risk — **{prob:.2f}** (Yüksək sağ qalma ehtimalı)")
    elif prob >= 0.75:
        st.warning(f"🟡 Orta Risk — **{prob:.2f}** (Orta sağ qalma ehtimalı)")
    else:
        st.error(f"🔴 Yüksək Risk — **{prob:.2f}** (Aşağı sağ qalma ehtimalı)")


    st.write("---")


# -----------------------------------------------------------
# ALWAYS-VISIBLE RESULTS (ACCORDION)
# -----------------------------------------------------------
st.markdown("---")

with st.expander("📊 Confusion Matrix"):
    st.image("images/xgb_confusion_matrix.png", width=520)

with st.expander("📈 ROC Curve"):
    st.image("images/xgb_roc_curve.png", width=520)

# with st.expander("📉 Calibration Curve"):
#     st.image("images/calibration_curve.png", width=520)

with st.expander("🔥 Feature Importance (XGBoost)"):
    st.image("images/xgb_feature_importance_top10.png", width=520)

with st.expander("🧠 SHAP Summary Plot"):
    st.image("images/xgb_shap_summary.png", width=520)


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.caption("Developed by Etibar Vazirov — Calibrated ML · Clinical AI · 2025")










