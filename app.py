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
# Top 5 features from PFI (reduced input interface)
# -----------------------------------------------------------
TOP_FEATURES = [
    "T Stage",
    "Reginol Node Positive",
    "Tumor Size",
    "N Stage",
    "Regional Node Examined"
]

# -----------------------------------------------------------
# GraphSAGE model definition
# -----------------------------------------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# -----------------------------------------------------------
# Load model
# -----------------------------------------------------------
in_dim = len(FEATURES)
hid_dim = 64
out_dim = len(STAGE_LABELS)

model = GraphSAGE(in_dim, hid_dim, out_dim)
model.load_state_dict(torch.load("sage_model.pt", map_location="cpu"))
model.eval()

edge_index = torch.tensor([[0], [0]], dtype=torch.long)

# -----------------------------------------------------------
# STREAMLIT PAGE SETTINGS
# -----------------------------------------------------------
st.set_page_config(page_title="Cancer Stage Prediction", page_icon="🩺")

# -----------------------------------------------------------
# HEADER & INTRODUCTION
# -----------------------------------------------------------
st.title("🩺 Breast Cancer Stage Prediction (Graph Neural Network)")
st.write("""
Bu tətbiq döş xərçənginin klinik göstəricilərinə əsaslanaraq **IIA–IIIC mərhələlərini**
təyin edən **GraphSAGE** təlimli modelindən istifadə edir.

Sistem yalnız ən vacib 5 klinik parametrdən istifadə edir 
(PFI — Permutation Feature Importance nəticələrinə əsasən):

- **T Stage**
- **Reginol Node Positive**
- **Tumor Size**
- **N Stage**
- **Regional Node Examined**

Bu göstəricilərin əsasında model mərhələni proqnozlaşdırır.
""")

st.markdown("---")

# -----------------------------------------------------------
# FORM INPUTS (DROP-DOWN + MANUAL)
# -----------------------------------------------------------

st.subheader("📥 Dəyərləri daxil edin")

input_data = {}

col1, col2 = st.columns(2)

# ▪▪▪ FEATURE: T Stage
with col1:
    t_options = ["T1", "T2", "T3", "T4"]
    t_stage = st.selectbox("T Stage", t_options)
    input_data["T Stage"] = t_stage

# ▪▪▪ FEATURE: Reginol Node Positive
with col2:
    rnp = st.number_input("Reginol Node Positive", min_value=0, max_value=30, step=1)
    input_data["Reginol Node Positive"] = str(rnp)

# ▪▪▪ FEATURE: Tumor Size
with col1:
    ts = st.number_input("Tumor Size (mm)", min_value=1, max_value=200, step=1)
    input_data["Tumor Size"] = str(ts)

# ▪▪▪ FEATURE: N Stage
with col2:
    n_options = ["N1", "N2", "N3"]
    n_stage = st.selectbox("N Stage", n_options)
    input_data["N Stage"] = n_stage

# ▪▪▪ FEATURE: Regional Node Examined
with col1:
    rne = st.number_input("Regional Node Examined", min_value=0, max_value=60, step=1)
    input_data["Regional Node Examined"] = str(rne)


st.markdown("---")

# -----------------------------------------------------------
# PREDICTION BUTTON
# -----------------------------------------------------------

if st.button("🔮 Predict Stage"):

    # 1) Check if user filled inputs
    if any(v == "" for v in input_data.values()):
        st.error("⚠️ Zəhmət olmasa bütün parametrləri daxil edin.")
    else:
        # 2) Fill missing features with zeros
        full_input = {feat: "0" for feat in FEATURES}
        full_input.update(input_data)

        # 3) Preprocess
        x_arr = preprocess_input(full_input, FEATURES)
        x_tensor = torch.tensor(x_arr, dtype=torch.float).unsqueeze(0)

        # 4) Run inference
        with torch.no_grad():
            out = model(x_tensor, edge_index)
            pred_idx = int(torch.argmax(out, dim=1).item())

        pred_stage = STAGE_LABELS[str(pred_idx)]

        st.success(f"🎯 **Proqnozlaşdırılan mərhələ: {pred_stage}**")

        st.markdown("---")

        # -----------------------------------------------------------
        # ACCORDIONS: RESULTS, TABLES & XAI
        # -----------------------------------------------------------
        with st.expander("📊 Model Performance"):
            st.image("images/model_comparison_sage_gat.png", width=550)

        with st.expander("📉 Confusion Matrix"):
            st.image("images/confusion_matrix_sage.png", width=550)

        with st.expander("📄 Classification Report"):
            st.image("images/classification_report_sage.png", width=550)

        with st.expander("🧠 Explainability (PFI — Global XAI)"):
            st.image("images/pfi_global_importance_sage.png", width=550)

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.caption("Developed by Etibar Vazirov · Graph Neural Networks · Explainable AI · 2025")
