import numpy as np

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
    Automatically aligns input keys with encoder keys (fixes spacing issues).
    """

    # ---------------------------
    # 1) AUTO-MAP FEATURE NAMES
    # ---------------------------
    encoder_keys = list(encoders.keys())
    corrected = {}

    for feat, val in data.items():
        # try exact match
        if feat in encoder_keys:
            corrected[feat] = val
            continue

        # try match ignoring spaces
        matched = None
        for ek in encoder_keys:
            if ek.strip().lower() == feat.strip().lower():
                matched = ek
                break

        if matched:
            corrected[matched] = val
        else:
            corrected[feat] = val   # numerical fields or non-encoded fields

    data = corrected


    # ---------------------------
    # 2) BUILD FEATURE VECTOR
    # ---------------------------
    row = []

    num_cols = ["Tumor Size", "Regional Node Positive"]
    cat_cols = list(encoders.keys())

    for feat in FEATURES:
        if feat in num_cols:
            row.append(float(data[feat]))
        else:
            encoder_key = feat
            # If encoder uses a different key (e.g., "T Stage "):
            for ek in encoder_keys:
                if ek.strip().lower() == feat.strip().lower():
                    encoder_key = ek
                    break

            row.append(encoders[encoder_key].transform([data[encoder_key]])[0])

    row = np.array(row).reshape(1, -1)

    # scale numericals
    idxs = [FEATURES.index(c) for c in num_cols]
    row[:, idxs] = scaler.transform(row[:, idxs])

    return row
