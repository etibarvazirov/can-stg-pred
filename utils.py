import numpy as np

def preprocess_input(user_input, FEATURES, encoders, scaler, num_cols):
    """
    user_input: dict
    FEATURES: full feature list (order matters)
    encoders: categorical encoders dictionary
    scaler: fitted StandardScaler
    num_cols: numerical feature names
    """

    row = []

    for feat in FEATURES:
        value = user_input.get(feat, "0")

        # Numeric columns
        if feat in num_cols:
            try:
                row.append(float(value))
            except:
                row.append(0.0)

        # Categorical columns
        else:
            encoder = encoders[feat]
            try:
                encoded = encoder.transform([value])[0]
            except:
                encoded = encoder.transform([encoder.classes_[0]])[0]
            row.append(encoded)

    row = np.array(row).reshape(1, -1)

    # Scale numerical columns
    row[:, [FEATURES.index(c) for c in num_cols]] = scaler.transform(
        row[:, [FEATURES.index(c) for c in num_cols]]
    )

    return row
