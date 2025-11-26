import streamlit as st
import pickle
import numpy as np
import torch
from utils import preprocess_input

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Survival Prediction",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------------------------------------
# LOAD ARTIFACTS
# -----------------------------------------------------------
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------------------------------------
# SESSION STATE FOR PRESETS
# -----------------------------------------------------------
if "preset" not in st.session_state:
    st.session_state.preset = None

# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown("""
<div style="background-color:#DFF5E3; padding:18px; border-radius:10px; 
            text-align:center; border:1px solid #B7E4C7; margin-bottom:15px;">
    <h1 style="color:#0C513F; margin:0; font-size:26px;">
        🩺 Breast Cancer Survival Prediction (XGBoost)
    </h1>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# DESCRIPTION BOX
# -----------------------------------------------------------
st.markdown("""
<div style="background-color:#F2FBF5; padding:15px; border-radius:10px;
            border-left:4px solid #66C2A5; font-size:16px; margin-bottom:20px;">
Bu sistem SEER döş xərçəngi məlumatları üzrə öyrədilmiş <b>XGBoost</b> modelindən istifadə edərək 
xəstənin <b>5 illik sağ qalma ehtimalını</b> proqnozlaşdırır.

Model klinik xüsusiyyətlərdən (T Stage, N Stage, Tumor Size, hormonal status və s.) istifadə edir
və nəticələri 3 risk səviyyəsi üzrə təqdim edir:
<br><br>
🟢 <b>Aşağı Risk</b> (≥ 0.92)  — yüksək sağ qalma ehtimalı  
🟡 <b>Orta Risk</b> (0.75 – 0.92)  
🔴 <b>Yüksək Risk</b> (< 0.75)  
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# PRESET BUTTONS  (must come BEFORE widgets)
# -----------------------------------------------------------
st.subheader("📌 Hazır nümunələr (Presets)")

colA, colB, colC = st.columns(3)

if colA.button("🟢 Low Risk"):
    st.session_state.preset = "low"
    st.rerun()

if colB.button("🟡 Medium Risk"):
    st.session_state.preset = "medium"
    st.rerun()

if colC.button("🔴 High Risk"):
    st.session_state.preset = "high"
    st.rerun()

# -----------------------------------------------------------
# PRESET VALUES (applied BEFORE widgets)
# -----------------------------------------------------------
# DEFAULT VALUES
default_values = {
    "tumor_size": 20,
    "rnp": 0,
    "t_stage": "T1",
    "n_stage": "N1",
    "diff": "Moderately differentiated",
    "grade": "2",
    "er": "Positive",
    "pr": "Positive",
    "race": "White"
}

# APPLY PRESET CHOICE
if st.session_state.preset == "low":
    default_values = {
        "tumor_size": 8,
        "rnp": 0,
        "t_stage": "T1",
        "n_stage": "N1",
        "diff": "Well differentiated",
        "grade": "1",
        "er": "Positive",
        "pr": "Positive",
        "race": "White"
    }

elif st.session_state.preset == "medium":
    default_values = {
        "tumor_size": 38,
        "rnp": 4,
        "t_stage": "T2",
        "n_stage": "N2",
        "diff": "Moderately differentiated",
        "grade": "2",
        "er": "Positive",
        "pr": "Negative",
        "race": "Other"
    }

elif st.session_state.preset == "high":
    default_values = {
        "tumor_size": 90,
        "rnp": 12,
        "t_stage": "T3",
        "n_stage": "N3",
        "diff": "Poorly differentiated",
        "grade": "3",
        "er": "Negative",
        "pr": "Negative",
        "race": "Black"
    }

# -----------------------------------------------------------
# INPUT FORM (widgets use preset defaults)
# -----------------------------------------------------------
st.subheader("📥 Dəyərləri daxil edin")

user_input = {}
col1, col2 = st.columns(2)

with col1:
    user_input["Tumor Size"] = st.number_input(
        "Tumor Size (mm)", 1, 200, default_values["tumor_size"]
    )

with col2:
    user_input["Reginol Node Positive"] = st.number_input(
        "Reginol Node Positive", 0, 30, default_values["rnp"]
    )

with col1:
    user_input["T Stage"] = st.selectbox(
        "T Stage",
        ["T1", "T2", "T3", "T4"],
        index=["T1","T2","T3","T4"].index(default_values["t_stage"])
    )

with col2:
    user_input["N Stage"] = st.selectbox(
        "N Stage",
        ["N1","N2","N3"],
        index=["N1","N2","N3"].index(default_values["n_stage"])
    )

with col1:
    user_input["differentiate"] = st.selectbox(
        "Differentiate",
        ["Poorly differentiated", "Moderately differentiated",
         "Well differentiated", "Undifferentiated"],
        index=[
            "Poorly differentiated",
            "Moderately differentiated",
            "Well differentiated",
            "Undifferentiated"
        ].index(default_values["diff"])
    )

with col2:
    user_input["Grade"] = st.selectbox(
        "Grade",
        ["1","2","3"," anaplastic; Grade IV"],
        index=["1","2","3"," anaplastic; Grade IV"].index(default_values["grade"])
    )

with col1:
    user_input["Estrogen Status"] = st.selectbox(
        "Estrogen Status",
        ["Positive","Negative"],
        index=["Positive","Negative"].index(default_values["er"])
    )

with col2:
    user_input["Progesterone Status"] = st.selectbox(
        "Progesterone Status",
        ["Positive","Negative"],
        index=["Positive","Negative"].index(default_values["pr"])
    )

with col1:
    user_input["Race"] = st.selectbox(
        "Race",
        ["White","Black","Other"],
        index=["White","Black","Other"].index(default_values["race"])
    )

st.markdown("---")

# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------
if st.button("🔮 Proqnoz et"):

    X = preprocess_input(user_input, encoders, scaler)
    prob = model.predict_proba(X)[0][1]   # survival probability

    # 3-LEVEL RISK SYSTEM
    if prob >= 0.92:
        st.success(f"🟢 Aşağı Risk — **{prob:.2f}** (Yüksək sağ qalma ehtimalı)")
    elif prob >= 0.75:
        st.warning(f"🟡 Orta Risk — **{prob:.2f}**")
    else:
        st.error(f"🔴 Yüksək Risk — **{prob:.2f}**")

st.markdown("---")

# -----------------------------------------------------------
# ACCORDIONS FOR RESULTS
# -----------------------------------------------------------
with st.expander("📊 Confusion Matrix"):
    st.image("images/xgb_confusion_matrix.png", width=520)

with st.expander("📈 Feature Importance (Top 10)"):
    st.image("images/xgb_feature_importance_top10.png", width=520)

with st.expander("🧠 SHAP Summary Plot"):
    st.image("images/xgb_shap_summary.png", width=520)

# with st.expander("🧠 SHAP Summary Plot"):
#     st.image("images/xgb_shap_summary.png", width=520)

st.markdown("---")
st.caption("Developed by Etibar Vazirov · XGBoost · Explainable AI · 2025")

