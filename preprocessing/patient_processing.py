import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def process_patient_dataset(file_path, reference_columns=None):

    df = pd.read_csv(file_path)

    # Important columns
    important = [
        "age",
        "anchor_age",
        "gender",
        "disease",
        "icd_code",
        "drug",
        "medication"
    ]

    selected_columns = [c for c in df.columns if c.lower() in important]

    if len(selected_columns) == 0:
        raise ValueError("No matching patient columns found.")

    df = df[selected_columns]

    # Encode categorical data
    df = pd.get_dummies(df)

    df = df.fillna(0)

    # Align columns across datasets
    if reference_columns is not None:
        df = df.reindex(columns=reference_columns, fill_value=0)

    scaler = StandardScaler()

    features = scaler.fit_transform(df)

    features = torch.tensor(features, dtype=torch.float)

    return features, df.columns