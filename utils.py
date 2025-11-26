import numpy as np

def preprocess_input(user_input, encoders, scaler):
    # categorical
    t_val = encoders["T Stage"].transform([user_input["T Stage"]])[0]
    n_val = encoders["N Stage"].transform([user_input["N Stage"]])[0]

    # numeric
    numeric_raw = np.array([
        float(user_input["Tumor Size"]),
        float(user_input["Reginol Node Positive"]),
        float(user_input["Regional Node Examined"]),
    ]).reshape(1, -1)

    numeric_scaled = scaler.transform(numeric_raw)[0]

    row = np.array([
        t_val,
        n_val,
        numeric_scaled[0],
        numeric_scaled[1],
        numeric_scaled[2]
    ]).reshape(1, -1)

    return row
