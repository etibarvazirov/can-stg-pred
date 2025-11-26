import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils import preprocess_input

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Survival Prediction", page_icon="🩺")

# -----------------------------------------------------------
# MEDICAL HEADER
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
        <h1 style="color:#0C513F; margin:0; font-size:26px;">
            🩺 Breast Cancer Survival Prediction (XGBoost Model)
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
    color:#0C513F;
">
Bu tətbiq döş xərçəngi xəstələrinin klinik göstəricilərinə əsaslanaraq **1 illik sağ qalma ehtimalını** 
proqnozlaşdırır. Model XGBoost alqoritmi ilə SEER məlumatlarına uyğun şəkildə öyrədilmişdir.

Proqnozda yalnız **çox vacib və klinik cəhətdən informativ** olan 9 göstəricidən istifadə olunur:

</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# TOP FEATURES (mini cards)
# -----------------------------------------------------------
TOP_FEATURES = {
    "Tumor Size": "Şişin faktiki ölçüsü (mm).",
    "Reginol Node Positive": "Xərçəng tapılan limfa düyünlərinin sayı.",
    "T Stage ": "Şişin ilkin T kateqoriyası (ölçü + yayılma dərinliyi).",
    "N Stage": "Limfa düyünlərinə yayılma dərəcəsi.",
    "differentiate": "Hüceyrələrin nə dərəcədə normal hüceyrəyə bənzəməsi.",
    "Grade": "Şişin hüceyrə dərəcəsi (I–IV).",
    "Estrogen Status": "ER pozitiv/negativ.",
    "Progesterone Status": "PR pozitiv/negativ.",
    "Race": "Xəstənin irqi."
}

st.markdown("<h4>📌 Proqnoz üçün istifadə olunan klinik göstəricilər</h4>", unsafe_allow_html=True)

for k, v in TOP_FEATURES.items():
    st.markdown(
        f"""
        <div style="
            background-color:#E9F7EF;
            padding:12px;
            margin-bottom:8px;
            border-radius:8px;
            border-left:4px solid #2ECC71;
        ">
            <b style="color:#0C513F; font-size:16px;">{k}</b><br>
            <span style="color:#1B4332; font-size:14px;">{v}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# -----------------------------------------------------------
# LOAD MODEL + ENCODERS + SCALER
# -----------------------------------------------------------
model = joblib.load("xgb_model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------------------------------------
# INPUT FORM
# -----------------------------------------------------------
st.subheader("📥 Dəyərləri daxil edin")

user_input = {}
col1, col2 = st.columns(2)

# Numerical input
with col1:
    user_input["Tumor Size"] = st.number_input("Tumor Size (mm)", 1, 200, 20)

with col2:
    user_input["Reginol Node Positive"] = st.number_input("Reginol Node Positive", 0, 30, 0)

# Categorical input
with col1:
    user_input["T Stage "] = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])

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

    prob = model.predict_proba(X)[0][1]

    if prob >= 0.5:
        st.success(f"🟢 Xəstənin sağ qalma ehtimalı yüksəkdir — **{prob:.2f}**")
    else:
        st.error(f"🔴 Xəstənin sağ qalma ehtimalı aşağıdır — **{prob:.2f}**")

st.markdown("---")

# -----------------------------------------------------------
# ACCORDIONS (SHAP + Feature Importance)
# -----------------------------------------------------------
with st.expander("📊 XGBoost Feature Importance"):
    st.image("images/xgb_feature_importance_top10.png", width=600)

with st.expander("🧠 SHAP Summary Plot"):
    st.image("images/xgb_shap_summary.png", width=600)

# with st.expander("🧬 SHAP Beeswarm Plot"):
#     st.image("images/xgb_shap_beeswarm.png", width=600)

st.markdown("---")
st.caption("Developed by ________ · XGBoost · Explainable AI · 2025")

