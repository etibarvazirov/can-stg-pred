import streamlit as st
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import numpy as np
import json

from utils import preprocess_input

# -----------------------------------------------------------
# Load metadata
# -----------------------------------------------------------
with open("feature_info.json", "r") as f:
    INFO = json.load(f)

FEATURES = INFO["features"]
STAGE_LABELS = INFO["stage_labels"]

# -----------------------------------------------------------
# Top 5 most important clinical features
# -----------------------------------------------------------
TOP_FEATURES = [
    "T Stage",
    "Reginol Node Positive",
    "Tumor Size",
    "N Stage",
    "Regional Node Examined"
]

FEATURE_DESCRIPTIONS = {
    "T Stage": "Şişin ilkin ölçüsü və toxumalara yayılma dərəcəsi.",
    "Reginol Node Positive": "Bölgədə xərçəng hüceyrələri tapılan limfa düyünlərinin sayı.",
    "Tumor Size": "Şişin real ölçüsü (mm). Böyük ölçü daha yüksək mərhələyə işarədir.",
    "N Stage": "Şişin limfa düyünlərinə yayılma dərəcəsi.",
    "Regional Node Examined": "Yoxlanılan limfa düyünlərinin ümumi sayı."
}

# -----------------------------------------------------------
# GraphSAGE model
# -----------------------------------------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# -----------------------------------------------------------
# Load model
# -----------------------------------------------------------
model = GraphSAGE(len(FEATURES), 64, len(STAGE_LABELS))
model.load_state_dict(torch.load("sage_model.pt", map_location="cpu"))
model.eval()

edge_index = torch.tensor([[0], [0]], dtype=torch.long)

# -----------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Stage Prediction", page_icon="🩺")

# -----------------------------------------------------------
# HEADER — Stylish Clinical Navbar
# -----------------------------------------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0d6efd, #228be6);
        padding: 18px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 15px;">
        <h1 style="color: white; margin: 0; font-size: 26px;">
            🩺 Breast Cancer Stage Prediction (Graph Neural Network)
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------
# Introduction with friendly clinical style
# -----------------------------------------------------------
st.write("""
Bu tətbiq döş xərçənginin klinik məlumatlarına əsaslanaraq **IIA–IIIC** mərhələlərini
proqnozlaşdıran **GraphSAGE** əsaslı süni intellekt modelidir.

Model yalnız ən güclü təsir göstərən 5 klinik göstəricidən istifadə edir 
(**Permutation Feature Importance** nəticəsinə əsaslanır):

""")

for feat in TOP_FEATURES:
    st.markdown(f"**• {feat}** — *{FEATURE_DESCRIPTIONS[feat]}*")

st.markdown("---")

# -----------------------------------------------------------
# INPUT FORM
# -----------------------------------------------------------
st.subheader("📥 Kliniki parametrləri daxil edin")

input_data = {}
col1, col2 = st.columns(2)

# -------------------------------
# 1. T Stage (dropdown)
# -------------------------------
with col1:
    t_stage = st.selectbox(
        "T Stage",
        ["T1", "T2", "T3", "T4"],
        help=FEATURE_DESCRIPTIONS["T Stage"]
    )
    input_data["T Stage"] = t_stage

# -------------------------------
# 2. Reginol Node Positive
# -------------------------------
with col2:
    rnp = st.number_input(
        "Reginol Node Positive",
        0, 30, help=FEATURE_DESCRIPTIONS["Reginol Node Positive"]
    )
    input_data["Reginol Node Positive"] = str(rnp)

# -------------------------------
# 3. Tumor Size
# -------------------------------
with col1:
    ts = st.number_input(
        "Tumor Size (mm)",
        1, 200,
        help=FEATURE_DESCRIPTIONS["Tumor Size"]
    )
    input_data["Tumor Size"] = str(ts)

# -------------------------------
# 4. N Stage
# -------------------------------
with col2:
    n_stage = st.selectbox(
        "N Stage",
        ["N1", "N2", "N3"],
        help=FEATURE_DESCRIPTIONS["N Stage"]
    )
    input_data["N Stage"] = n_stage

# -------------------------------
# 5. Regional Node Examined
# -------------------------------
with col1:
    rne = st.number_input(
        "Regional Node Examined",
        0, 60,
        help=FEATURE_DESCRIPTIONS["Regional Node Examined"]
    )
    input_data["Regional Node Examined"] = str(rne)

st.markdown("---")

# -----------------------------------------------------------
# PREDICTION BUTTON
# -----------------------------------------------------------
if st.button("🔮 Proqnoz et"):

    if any(v == "" for v in input_data.values()):
        st.error("⚠️ Zəhmət olmasa bütün zəruri sahələri doldurun.")
    else:
        # Expand to full 16 features
        full_input = {feat: "0" for feat in FEATURES}
        full_input.update(input_data)

        # Preprocess
        x_arr = preprocess_input(full_input, FEATURES)
        x_tensor = torch.tensor(x_arr, dtype=torch.float).unsqueeze(0)

        # Predict
        with torch.no_grad():
            out = model(x_tensor, edge_index)
            pred_idx = int(out.argmax(dim=1).item())

        pred_stage = STAGE_LABELS[str(pred_idx)]

        st.success(f"🎯 **Proqnozlaşdırılan mərhələ: {pred_stage}**")

        st.markdown("---")

        # -----------------------------------------------------------
        # ACCORDIONS WITH EXPLANATION
        # -----------------------------------------------------------
        with st.expander("📊 Model Performance"):
            st.write("Bu qrafik GraphSAGE və GAT modellərinin nəticələrini müqayisə edir.")
            st.image("images/model_comparison_sage_gat.png", width=550)

        with st.expander("📉 Confusion Matrix"):
            st.write("Hər bir mərhələ üzrə modelin düzgün və yanlış təsnifatlarını göstərir.")
            st.image("images/confusion_matrix_sage.png", width=550)

        with st.expander("📄 Classification Report"):
            st.write("Hər mərhələ üçün Precision, Recall və F1-score dəyərlərini göstərir.")
            st.image("images/classification_report_sage.png", width=550)

        with st.expander("🧠 Explainability (PFI — Global XAI)"):
            st.write("Bu qrafik modelin qərarlarına ən çox təsir edən klinik göstəriciləri göstərir.")
            st.image("images/pfi_global_importance_sage.png", width=550)

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.caption("Developed by Etibar Vazirov · Graph Neural Networks · Explainable AI · 2025")
