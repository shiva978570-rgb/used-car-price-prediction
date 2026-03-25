import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

model = pickle.load(open(os.path.join(PROJECT_DIR, "models", "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(PROJECT_DIR, "models", "scaler.pkl"), "rb"))
feature_columns = pickle.load(open(os.path.join(PROJECT_DIR, "models", "features.pkl"), "rb"))


def predict_price(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    return prediction[0]


if __name__ == "__main__":
    sample = {
        "year": 2018,
        "km_driven": 50000,
        "owner": 1,
        "fuel": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual"
    }

    price = predict_price(sample)
    print("Predicted Price:", price)