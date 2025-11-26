import numpy as np
import json

# Load feature order
with open("feature_info.json", "r") as f:
    INFO = json.load(f)

FEATURES = INFO["features"]

# Correct encoders (SEER dataset)
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

def preprocess_input(user_input, feature_names):

    # Final feature array to send into the model
    x = []

    print("DEBUG: ENCODERS KEYS =", list(ENCODERS.keys()))
    print("DEBUG: USER INPUT KEYS =", list(user_input.keys()))


    for feat in feature_names:

        # If this feature was NOT provided in Streamlit → 0
        if feat not in user_input:
            x.append(0.0)
            continue

        val = user_input[feat]

        # Categorical
        if feat in ENCODERS:
            mapping = ENCODERS[feat]
            x.append(float(mapping.get(val, 0)))

        # Numeric
        else:
            try:
                x.append(float(val))
            except:
                x.append(0.0)

    return np.array(x, dtype=float)



