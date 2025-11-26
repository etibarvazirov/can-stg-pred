import numpy as np
import json

with open("feature_info.json", "r") as f:
    INFO = json.load(f)

FEATURES = INFO["features"]

# Encoders defined correctly
ENCODERS = {
    "T Stage": {"T1":0, "T2":1, "T3":2, "T4":3},
    "N Stage": {"N1":0, "N2":1, "N3":2},
    "Race": {"White":0, "Black":1, "Other":2},
    "Marital Status": {
        "Married":0, "Divorced":1, "Single":2, "Widowed":3, "Separated":4
    },
    "differentiate": {
        "Well differentiated":0,
        "Moderately differentiated":1,
        "Poorly differentiated":2,
        "Undifferentiated":3
    },
    "Grade": {"1":1, "2":2, "3":3, "anaplastic; Grade IV":4},
    "A Stage": {"Regional":0, "Distant":1},
    "Estrogen Status": {"Positive":1, "Negative":0},
    "Progesterone Status": {"Positive":1, "Negative":0},
    "Status": {"Alive":0, "Dead":1}
}

# Default categorical values
DEFAULT_CATEGORICAL = {
    "Race": "White",
    "Marital Status": "Married",
    "differentiate": "Moderately differentiated",
    "Grade": "2",
    "A Stage": "Regional",
    "Estrogen Status": "Positive",
    "Progesterone Status": "Positive",
    "Status": "Alive"
}

def preprocess_input(user_input, feature_names):

    x = []

    for feat in feature_names:

        # If provided by Streamlit
        if feat in user_input:
            val = user_input[feat]

        # Else if categorical → use default
        elif feat in DEFAULT_CATEGORICAL:
            val = DEFAULT_CATEGORICAL[feat]

        # Else numeric → fallback 0
        else:
            val = "0"

        # Now encode properly
        if feat in ENCODERS:
            mapping = ENCODERS[feat]
            x.append(float(mapping.get(val, 0)))
        else:
            try:
                x.append(float(val))
            except:
                x.append(0.0)

    return np.array(x, dtype=float)
