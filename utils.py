import numpy as np
import json

# Load metadata (feature order + stage labels)
with open("feature_info.json", "r") as f:
    INFO = json.load(f)

FEATURES = INFO["features"]

# ---------------------------------------------------------
# CATEGORY ENCODERS — EXACT from your dataset
# ---------------------------------------------------------

ENCODERS = {
    "Race": {
        "White": 0,
        "Black": 1,
        "Other": 2
    },

    "Marital Status": {
        "Married": 0,
        "Divorced": 1,
        "Single": 2,     # cleaned version
        "Widowed": 3,
        "Separated": 4
    },

    "T Stage": {
        "T1": 0,
        "T2": 1,
        "T3": 2,
        "T4": 3
    },

    "N Stage": {
        "N1": 1,
        "N2": 2,
        "N3": 3
    },

    "differentiate": {
        "Well differentiated": 0,
        "Moderately differentiated": 1,
        "Poorly differentiated": 2,
        "Undifferentiated": 3
    },

    "Grade": {
        "1": 1,
        "2": 2,
        "3": 3,
        "anaplastic; Grade IV": 4
    },

    "A Stage": {
        "Regional": 0,
        "Distant": 1
    },

    "Estrogen Status": {
        "Positive": 1,
        "Negative": 0
    },

    "Progesterone Status": {
        "Positive": 1,
        "Negative": 0
    },

    "Status": {
        "Alive": 0,
        "Dead": 1
    }
}

# ---------------------------------------------------------
# Preprocessing function for Streamlit
# ---------------------------------------------------------

def preprocess_input(user_input: dict, feature_names: list):
    """
    Converts Streamlit form inputs into a numeric numpy array
    matching graph.x input feature order.
    """
    x = []

    for feat in feature_names:
        raw = user_input.get(feat, "").strip()

        # Handle empty
        if raw == "" or raw is None:
            x.append(0.0)
            continue

        # Clean spacing issues
        raw = raw.strip()
        raw = raw.replace("  ", " ")

        # If feature is categorical:
        if feat in ENCODERS:
            mapping = ENCODERS[feat]
            value = mapping.get(raw, 0)   # if unseen → 0
            x.append(float(value))
        else:
            # Numeric feature
            try:
                x.append(float(raw))
            except:
                x.append(0.0)

    return np.array(x, dtype=float)

