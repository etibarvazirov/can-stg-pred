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
# Top 5 most important features
# -----------------------------------------------------------
TOP_FEATURES = [
    "T Stage",
    "Reginol Node Positive",
    "Tumor Size",
    "N Stage",
    "Regional Node Examined"
]

# FEATURE_DESCRIPTIONS = {
#     "T Stage": "Şişin ilkin ölçüsü və toxumalara yayılma dərəcəsi.",
#     "Reginol Node Positive": "Xərçəng hüceyrəsi tapılan limfa düyünlərinin sayı.",
#     "Tumor Size": "Şişin faktiki ölçüsü (mm).",
#     "N Stage": "Limfa düyünlərinə yayılma dərəcəsi.",
#     "Regional Node Examined": "Yoxlanılan limfa düyünlərinin ümumi sayı."
# }

# -----------------------------------------------------------
# GraphSAGE Model
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

# Load trained model
model = GraphSAGE(len(FEATURES), 64, len(STAGE_LABELS))
model.load_state_dict(torch.load("sage_model.pt", map_location="cpu"))
model.eval()

edge_index = torch.tensor([[0], [0]], dtype=torch.long)

# -----------------------------------------------------------
# Streamlit Config
# -----------------------------------------------------------
st.set_page_config(page_title="Cancer Stage Prediction", page_icon="🩺")

# -----------------------------------------------------------
# HEADER (Medical pastel design)
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
            🩺 Breast Cancer Stage Prediction (Graph Neural Network)
        </h1>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# DESCRIPTION (Medical INFO box)
# -----------------------------------------------------------
st.markdown("""
<div style="
    background-color:#F2FBF5;
    padding:15px;
    border-radius:10px;
    border-left:4px solid #66C2A5;
    font-size:16px;
">

b>Döş xərçəngi mərhələsinin proqnozlaşdırılması</b> xəstənin klinik göstəricilərinə əsaslanan
AI sistemlərində mühüm addımdır. Bu tətbiq SEER məlumatlarından öyrədilmiş 
<b>GraphSAGE</b> modelindən istifadə edərək xərçəngin <b>IIA–IIIC</b> mərhələləri üzrə proqnoz verir.

Model, SEER məlumatlarında təqdim olunan “6th Stage” təsnifatına əsaslanaraq döş xərçənginin beş klinik mərhələsini — <b>IIA, IIB, IIIA, IIIB və IIIC</b> — proqnozlaşdırır. Bu mərhələlər xərçəngin erkən (IIA, IIB), orta (IIIA) və daha irəliləmiş (IIIB, IIIC) yayılma səviyyələrini əks etdirir.

Bu sistem yalnız ən vacib klinik göstəricilərdən istifadə edir (Permutation Feature Importance nəticələrinə əsaslanır):

Model yalnız ən vacib klinik göstəricilərdən istifadə edir:
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# FEATURE DESCRIPTIONS — styled medical mini-cards
# -----------------------------------------------------------

st.markdown("<h4 style='margin-top:15px;'>📌 Ən vacib klinik göstəricilər</h4>", unsafe_allow_html=True)

for feat in TOP_FEATURES:
    desc = FEATURE_DESCRIPTIONS[feat]
    st.markdown(
        f"""
        <div style="
            background-color:#E9F7EF;
            padding:12px;
            margin-bottom:8px;
            border-radius:8px;
            border-left:4px solid #2ECC71;
        ">
            <b style="color:#0C513F; font-size:16px;">{feat}</b><br>
            <span style="color:#1B4332; font-size:14px;">{desc}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")

# -----------------------------------------------------------
# INPUT FORM
# -----------------------------------------------------------
st.subheader("📥 Kliniki parametrləri daxil edin")

input_data = {}
col1, col2 = st.columns(2)

with col1:
    input_data["T Stage"] = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])

with col2:
    rnp = st.number_input("Reginol Node Positive", 0, 30)
    input_data["Reginol Node Positive"] = str(rnp)

with col1:
    ts = st.number_input("Tumor Size (mm)", 1, 200)
    input_data["Tumor Size"] = str(ts)

with col2:
    input_data["N Stage"] = st.selectbox("N Stage", ["N1", "N2", "N3"])

with col1:
    rne = st.number_input("Regional Node Examined", 0, 60)
    input_data["Regional Node Examined"] = str(rne)

st.markdown("---")

# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------
if st.button("🔮 Proqnoz et"):

    if any(v == "" for v in input_data.values()):
        st.error("⚠️ Zəhmət olmasa bütün sahələri doldurun.")
    else:
        full_input = {feat: "0" for feat in FEATURES}
        full_input.update(input_data)

        x = preprocess_input(full_input, FEATURES)
        x_tensor = torch.tensor(x, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            out = model(x_tensor, edge_index)
            pred_idx = int(out.argmax(dim=1).item())

        pred_stage = STAGE_LABELS[str(pred_idx)]
        st.success(f"🎯 **Proqnozlaşdırılan mərhələ: {pred_stage}**")

st.markdown("---")

# -----------------------------------------------------------
# ALWAYS VISIBLE ACCORDIONS
# -----------------------------------------------------------
with st.expander("📊 Model Performance"):
    st.write("GraphSAGE və GAT modellərinin performansının müqayisəsi.")
    st.image("images/model_comparison_sage_gat.png", width=550)

with st.expander("📉 Confusion Matrix"):
    st.write("Hər mərhələ üzrə düzgün və yanlış təsnifat dəyərləri.")
    st.image("images/confusion_matrix_sage.png", width=550)

with st.expander("📄 Classification Report"):
    st.write("Hər sinif üçün Precision, Recall və F1-score göstəriciləri.")
    st.image("images/classification_report_sage.png", width=550)

with st.expander("🧠 Explainability (PFI — Global XAI)"):
    st.write("Modelin qərarına ən çox təsir edən klinik göstəricilər.")
    st.image("images/pfi_global_importance_sage.png", width=550)

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")

