import numpy as np
import pandas as pd

def preprocess_input(user_input, encoders, scaler):
    """
    user_input: dict of raw form inputs
    returns: numpy array shaped for model
    """

    FEATURES = [
        "Tumor Size",
        "Reginol Node Positive",
        "T Stage ",
        "N Stage",
        "differentiate",
        "Grade",
        "Estrogen Status",
        "Progesterone Status",
        "Race"
    ]

    row = []

    for feat in FEATURES:
        val = user_input[feat]

        # Numerical
        if feat in ["Tumor Size", "Reginol Node Positive"]:
            row.append(float(val))

        # Categorical
        else:
            le = encoders[feat]
            row.append(le.transform([val])[0])

    # Convert to DataFrame
    df_row = pd.DataFrame([row], columns=FEATURES)

    # Scale numeric only
    df_row[["Tumor Size", "Reginol Node Positive"]] = scaler.transform(
        df_row[["Tumor Size", "Reginol Node Positive"]]
    )

    return df_row.values
