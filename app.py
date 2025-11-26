import streamlit as st
import pickle
import numpy as np
import json

from utils import preprocess_input

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Survival Prediction", page_icon="🩺")

# -----------------------------------------------------------
# LOAD ARTIFACTS
# -----------------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("feature_info.json", "r") as f:
    INFO = json.load(f)

FEATURES = INFO["features"]
num_cols = INFO["num_cols"]
cat_cols = INFO["cat_cols"]
THRESHOLD = INFO["threshold"]   # Youden J optimal threshold

# -----------------------------------------------------------
# HEADER (Medical Style)
# -----------------------------------------------------------
st.markdown("""
    <div style="
        background-color:#D8F3DC;
        padding:18px;
        border-radius:10px;
        text-align:center;
        border:1px solid #95D5B2;
        margin-bottom:15px;">
        <h1 style="color:#1B4332; margin:0;">
            🩺 Breast Cancer 5-Year Survival Prediction
        </h1>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# DESCRIPTION
# -----------------------------------------------------------
st.markdown("""
Bu tətbiq döş xərçəngi xəstələri üçün **5 illik sağ qalma ehtimalını** təxmin edir.
Model XGBoost əsasında hazırlanmışdır və SEER klinik məlumatları üzərində öyrədilmişdir.

Sistem aşağıdakı ən vacib klinik göstəricilərdən istifadə edir:
- **Yaş (Age)**
- **Şişin ölçüsü (Tumor Size)**
- **Limfa düyünləri (N Stage)**
- **Hormon statusu (Estrogen / Progesterone)**
- **Histoloji dərəcə (Grade)**

Proqnoz:  
**1 → Alive (yüksək sağ qalma ehtimalı)**  
**0 → Dead (yüksək risk)**  
""")

st.markdown("---")

# -----------------------------------------------------------
# USER INPUTS
# -----------------------------------------------------------
st.subheader("📥 Xəstə məlumatlarını daxil edin")

user_input = {}

for feat in FEATURES:

    if feat in num_cols:
        val = st.number_input(f"{feat}", value=0.0)
        user_input[feat] = val
    else:
        options = list(encoders[feat].classes_)
        val = st.selectbox(f"{feat}", options)
        user_input[feat] = val

st.markdown("---")

# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------
if st.button("🔮 Proqnoz et"):

    X = preprocess_input(user_input, FEATURES, encoders, scaler, num_cols)
    prob_alive = model.predict_proba(X)[0][1]

    pred = 1 if prob_alive >= THRESHOLD else 0

    if pred == 1:
        st.success(f"🌿 **Nəticə: Xəstənin sağ qalma ehtimalı yüksəkdir (Alive)**\n\nEhtimal: {prob_alive:.2f}")
    else:
        st.error(f"⚠️ **Nəticə: Yüksək risk (Dead)**\n\nSağ qalma ehtimalı: {prob_alive:.2f}")

    st.markdown("---")

    # -----------------------------------------------------------
    # FIGURES
    # -----------------------------------------------------------
    with st.expander("📊 Model Accuracy Comparison"):
        st.image("images/model_cv_accuracy.png")

    with st.expander("📉 Confusion Matrix (Optimized)"):
        st.image("images/xgb_confusion_matrix.png")

    with st.expander("📈 ROC Curve"):
        st.image("images/xgb_roc_curve.png")

    with st.expander("🧠 Feature Importance (Top-10)"):
        st.image("images/xgb_feature_importance_top10.png")

    with st.expander("🧬 SHAP Summary Plot"):
        st.image("images/xgb_shap_summary.png")

st.markdown("---")
st.caption("Developed by Etibar Vazirov · 2025 · Survival AI Model")
