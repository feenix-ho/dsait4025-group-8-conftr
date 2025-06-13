import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import os

def preprocess_german_data(input_path):
    # Load data
    df = pd.read_csv(input_path).iloc[:, 1:]

    # Remove id column
    if df.columns[0] == "":
        df = df.drop(columns=df.columns[0])

    # Replace "NA" with NaN
    df = df.replace("NA", np.nan)

    # Change target from categorical to numerical
    df['Risk'] = df['Risk'].map({'good': 0, 'bad': 1})
    y = df['Risk'].values
    X = df.drop(columns=['Risk'])

    # Derive categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(exclude=['object']).columns

    # One-hot encoding
    ohe = OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(X[cat_cols]) if len(cat_cols) > 0 else np.empty((len(X), 0))
    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    # Standardize numerical features (per-feature whitening as in the paper)
    X_num = X[num_cols].fillna(X[num_cols].mean())
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Combine into one dataset
    X_processed = np.concatenate([X_num_scaled, X_cat], axis=1)
    feature_names = list(num_cols) + list(cat_feature_names)
    df_out = pd.DataFrame(X_processed, columns=feature_names)
    df_out["Risk"] = y

    return df_out

if __name__ == "__main__":
    input_csv = os.path.join(os.path.dirname(__file__), "german_credit_original.csv")
    output_csv = os.path.join(os.path.dirname(__file__), "german_credit.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df_processed = preprocess_german_data(input_csv)
    df_processed.to_csv(output_csv, index=False)