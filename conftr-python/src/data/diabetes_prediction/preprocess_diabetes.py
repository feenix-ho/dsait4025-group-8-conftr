import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def preprocess_diabetes_data(input_path):
    # Load data
    df = pd.read_csv(input_path)

    # Change target from categorical to numerical
    df['CLASS'] = df['CLASS'].str.strip()
    df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})
    y = df['CLASS'].values

    # Drop target column and ID columns
    X = df.drop(columns=['ID', 'No_Pation', 'CLASS'])

    # Map Gender to numerical
    X['Gender'] = X['Gender'].map({'F': 0, 'M': 1})
    X['Gender'] = X['Gender'].fillna(-1).astype(int)

    # Derive numerical columns
    num_cols = X.select_dtypes(exclude=['object']).columns

    # Standardize numerical features (per-feature whitening as in the paper)
    X_num = X[num_cols].fillna(X[num_cols].mean())
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Combine into one dataset
    X_processed = X_num_scaled
    feature_names = list(num_cols)
    df_out = pd.DataFrame(X_processed, columns=feature_names)
    df_out["CLASS"] = y

    return df_out

if __name__ == "__main__":
    input_csv = os.path.join(os.path.dirname(__file__), "diabetes_original.csv")
    output_csv = os.path.join(os.path.dirname(__file__), "diabetes.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df_processed = preprocess_diabetes_data(input_csv)
    df_processed.to_csv(output_csv, index=False)
