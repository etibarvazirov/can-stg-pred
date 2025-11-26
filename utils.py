import numpy as np

# 9 feature model
FEATURES = [
    "Tumor Size",
    "Regional Node Positive",
    "T Stage",
    "N Stage",
    "differentiate",
    "Grade",
    "Estrogen Status",
    "Progesterone Status",
    "Race"
]


def preprocess_input(data, encoders, scaler):
    """
    Converts raw user input into model-ready numerical vector.
    
    Steps:
    1. Numeric features -> float
    2. Categorical features -> LabelEncoder transform
    3. StandardScaler normalization for numeric features
    4. Return numpy array shaped (1, 9)
    """

    row = []

    # Numerical columns
    num_cols = ["Tumor Size", "Regional Node Positive"]

    # Categorical columns
    cat_cols = [
        "T Stage", "N Stage", "differentiate", "Grade",
        "Estrogen Status", "Progesterone Status", "Race"
    ]

    # Build row in correct order
    for feat in FEATURES:
        if feat in num_cols:
            row.append(float(data[feat]))
        else:
            row.append(encoders[feat].transform([data[feat]])[0])

    row = np.array(row).reshape(1, -1)

    # Scale numerical columns
    idxs = [FEATURES.index(c) for c in num_cols]
    row[:, idxs] = scaler.transform(row[:, idxs])

    return row



