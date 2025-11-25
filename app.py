import streamlit as st
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import numpy as np
import json

from utils import preprocess_input

# -----------------------------------------------------------
# Load metadata from JSON
# -----------------------------------------------------------
with open("feature_info.json", "r") as f:
    INFO = json.load(f)

FEATURES = INFO["features"]
STAGE_LABELS = INFO["stage_labels"]

# -----------------------------------------------------------
# Define GraphSAGE model (must match training architecture)
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
# Load model weights
# -----------------------------------------------------------
in_dim = len(FEATURES)
hid_dim = 64
out_dim = len(STAGE_LABELS)

model = GraphSAGE(in_dim, hid_dim, out_dim)
model.load_state_dict(torch.load("sage_model.pt", map_location="cpu"))
model.eval()

# Dummy edge_index (prediction uses only node 0)
edge_index = torch.tensor([[0], [0]], dtype=torch.long)

# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------
st.set_page_config(page_title="Cancer Stage Prediction (GNN)", page_icon="🩺")

st.title("🩺 Breast Cancer Stage Prediction Using Graph Neural Networks")
st.write("This AI system predicts **cancer stage (IIA–IIIC)** using a trained GraphSAGE model.")

st.subheader("📝 Enter Patient Clinical Information")

# Collect user inputs
user_input = {}
cols = st.columns(2)

for i, feat in enumerate(FEATURES):
    with cols[i % 2]:
        user_input[feat] = st.text_input(f"{feat}:")

# -----------------------------------------------------------
# Prediction Button
# -----------------------------------------------------------
if st.button("🔮 Predict Stage"):
    try:
        # Preprocess input into numeric array
        x_arr = preprocess_input(user_input, FEATURES)
        x_tensor = torch.tensor(x_arr, dtype=torch.float).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            out = model(x_tensor, edge_index)
            pred_class = int(torch.argmax(out, dim=1).item())

        pred_stage = STAGE_LABELS[str(pred_class)]

        st.success(f"🎯 **Predicted Cancer Stage: {pred_stage}**")

        st.subheader("📊 Model Explanation (Global XAI — PFI)")
        st.info("The chart below shows which features globally influenced the model the most.")

        st.image("pfi_global_importance_sage.png", width=550)

    except Exception as e:
        st.error("⚠️ An error occurred during prediction.")
        st.text(str(e))

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.markdown("---")
st.caption("Developed by Etibar Vazirov · Graph Neural Networks · Explainable AI")


