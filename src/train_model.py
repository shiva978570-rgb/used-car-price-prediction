import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from preprocessing import load_data, preprocess_data, split_data, scale_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "Data", "car.csv")


def train():
    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)


    feature_columns = X_train.columns.tolist()


    X_train, X_test, scaler = scale_data(X_train, X_test)

    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    
    print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
    print("Random Forest R2:", r2_score(y_test, y_pred_rf))

    
    model_path = os.path.join(PROJECT_DIR, "models", "model.pkl")
    scaler_path = os.path.join(PROJECT_DIR, "models", "scaler.pkl")
    feature_path = os.path.join(PROJECT_DIR, "models", "features.pkl")

    
    pickle.dump(rf, open(model_path, "wb"))
    pickle.dump(scaler, open(scaler_path, "wb"))
    pickle.dump(feature_columns, open(feature_path, "wb"))

    print("Model saved successfully!")


if __name__ == "__main__":
    train()