import streamlit as st
import numpy as np
import pickle
from utils import preprocess_input

# -----------------------------------------------------------
# Load artifacts
# -----------------------------------------------------------
model = pickle.load(open("xgb_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Stage Prediction", page_icon="🩺")

# -----------------------------------------------------------
# HEADER (Pastel medical design)
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
            🩺 Breast Cancer Survival Prediction (XGBoost)
        </h1>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# DESCRIPTION
# -----------------------------------------------------------
st.markdown("""
<div style="
    background-color:#F2FBF5;
    padding:15px;
    border-radius:10px;
    border-left:4px solid #66C2A5;
    font-size:16px;
">
Bu sistem SEER real dünyadakı klinik məlumatları əsasında qurulmuş
<b>XGBoost</b> modelindən istifadə edərək xəstənin <b>yaşayıb-yaşamayacağını</b> proqnozlaşdırır.

Model yalnız ən vacib 5 klinik göstəricini istifadə edir:
<ul>
<li><b>T Stage</b></li>
<li><b>N Stage</b></li>
<li><b>Tumor Size</b></li>
<li><b>Reginol Node Positive</b></li>
<li><b>Regional Node Examined</b></li>
</ul>

Bu göstəricilər döş xərçənginin lokal və regional yayılmasını əks etdirir və xəstənin sağ qalma ehtimalı ilə sıx bağlıdır.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------
# USER INPUT FORM
# -----------------------------------------------------------
st.subheader("📥 Kliniki göstəriciləri daxil edin")

col1, col2 = st.columns(2)

user_input = {}

with col1:
    user_input["T Stage"] = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])

with col2:
    user_input["N Stage"] = st.selectbox("N Stage", ["N1", "N2", "N3"])

with col1:
    user_input["Tumor Size"] = st.number_input("Tumor Size (mm)", 1, 200)

with col2:
    user_input["Reginol Node Positive"] = st.number_input("Reginol Node Positive", 0, 30)

user_input["Regional Node Examined"] = st.number_input("Regional Node Examined", 0, 60)

st.markdown("---")

# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------
if st.button("🔮 Proqnoz et"):

    try:
        X = preprocess_input(user_input, encoders, scaler)
        prob_survival = model.predict_proba(X)[0][1]

        if prob_survival >= 0.5:
            st.success(f"🎯 Xəstənin sağ qalma ehtimalı yüksəkdir: **{prob_survival:.2f}**")
        else:
            st.error(f"⚠️ Sağ qalma ehtimalı aşağıdır: **{prob_survival:.2f}**")

        st.write("### 🔍 Modelə daxil olan feature vektoru:")
        st.write(X)

    except Exception as e:
        st.error(f"Xəta baş verdi: {e}")

st.markdown("---")

# -----------------------------------------------------------
# DIAGRAMS SECTION — Always visible
# -----------------------------------------------------------
with st.expander("📊 Model Performance (Confusion Matrix)"):
    st.image("images/xgb_confusion_matrix.png", width=550)
    st.write("Bu xəritə modelin düzgün və yanlış təsnifat etdiyi nümunələrin bölgüsünü göstərir.")

with st.expander("📈 ROC Curve"):
    st.image("images/xgb_roc_curve.png", width=550)
    st.write("ROC əyrisi modelin müxtəlif threshold-larda ayrıcılıq gücünü göstərir.")

with st.expander("📉 Feature Importance"):
    st.image("images/xgb_feature_importance_top10.png", width=550)
    st.write("XGBoost modelinə ən çox təsir edən klinik göstəricilər.")

with st.expander("🧠 SHAP Summary Plot (Global Explainability)"):
    st.image("images/xgb_shap_summary.png", width=550)
    st.write("Bu SHAP qrafiki modelin ümumi qərarlarına ən çox təsir edən xüsusiyyətləri göstərir.")

st.markdown("---")

st.caption("Developed by ... · XGBoost · Explainable AI · 2025")


