import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    if 'name' in df.columns:
        df.drop('name', axis=1, inplace=True)

    
    df.dropna(inplace=True)

    
    df = pd.get_dummies(df, drop_first=True)

    return df


def split_data(df):
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler