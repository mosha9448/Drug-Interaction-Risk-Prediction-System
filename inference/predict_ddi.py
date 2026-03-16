import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np

from models.mkmgcn_model import MKMGCN
from preprocessing.graph_construction import build_ddi_graph
from preprocessing.smiles_processing import process_smiles


print("Starting DDI prediction...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Load patient dataset
# --------------------------------------------------

patient_df = pd.read_csv("patient/patient_drug_mapped.csv")


# --------------------------------------------------
# Load drug datasets
# --------------------------------------------------

ddi = pd.read_csv("data/drug_interactions.csv")
smiles = pd.read_csv("data/drug_smiles_clean.csv")

ddi = ddi.rename(columns={
    "drug_id": "drugA",
    "interacting_drug_id": "drugB"
})


# --------------------------------------------------
# Merge SMILES
# --------------------------------------------------

smilesA = smiles.rename(columns={
    "drug_id": "drugA",
    "smiles": "smilesA"
})

smilesB = smiles.rename(columns={
    "drug_id": "drugB",
    "smiles": "smilesB"
})

df = ddi.merge(smilesA, on="drugA", how="left")
df = df.merge(smilesB, on="drugB", how="left")

df = df.dropna(subset=["smilesA", "smilesB"])


# --------------------------------------------------
# Process SMILES
# --------------------------------------------------

smiles_df = pd.DataFrame({
    "drugA": df["drugA"],
    "drugB": df["drugB"],
    "smilesA": df["smilesA"],
    "smilesB": df["smilesB"]
})

print("Processing SMILES...")

drug_mols, drug_features = process_smiles(smiles_df)


# --------------------------------------------------
# Build DDI graph
# --------------------------------------------------

print("Building DDI graph...")

edge_index, drug_to_idx = build_ddi_graph(df)

num_nodes = len(drug_to_idx)

print("Total drugs:", num_nodes)


# --------------------------------------------------
# Build feature matrix
# --------------------------------------------------

feature_dim = len(next(iter(drug_features.values())))
feature_matrix = np.zeros((num_nodes, feature_dim))

for drug, idx in drug_to_idx.items():
    feature_matrix[idx] = drug_features[drug]

x = torch.tensor(feature_matrix).float().to(device)
edge_index = edge_index.to(device)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = MKMGCN(num_nodes).to(device)

model.load_state_dict(torch.load("models/ddi_model.pth", map_location=device))

model.eval()

print("Model loaded.")


# --------------------------------------------------
# Compute embeddings
# --------------------------------------------------

with torch.no_grad():
    embeddings, _ = model(x, edge_index)


# --------------------------------------------------
# Disease encoding
# --------------------------------------------------

disease_map = {
    "Diabetes":0,
    "Hypertension":1,
    "Cancer":2,
    "Stroke":3,
    "Asthma":4,
    "Depression":5
}


# --------------------------------------------------
# Predict patient drug interaction risk
# --------------------------------------------------

results = []

for _, row in patient_df.iterrows():

    drugA = row["drugA_id"]
    drugB = row["drugB_id"]

    if drugA not in drug_to_idx or drugB not in drug_to_idx:
        continue

    idxA = drug_to_idx[drugA]
    idxB = drug_to_idx[drugB]

    embA = embeddings[idxA].unsqueeze(0)
    embB = embeddings[idxB].unsqueeze(0)

    # -----------------------------
    # Patient features
    # -----------------------------

    age = torch.tensor([[row["age"]/100]]).float().to(device)

    gender_val = 1 if row["gender"] == "Male" else 0
    gender = torch.tensor([[gender_val]]).float().to(device)

    disease_id = disease_map.get(row["disease"], 0)
    disease = torch.tensor([[disease_id]]).float().to(device)

    patient_features = torch.cat([age, gender, disease], dim=1)

    # -----------------------------

    risk = model.detector(embA, embB, patient_features).item()

    if risk > 0.7:
        level = "Major"
    elif risk > 0.4:
        level = "Moderate"
    else:
        level = "minor"

    results.append([
        row["patient_id"],
        drugA,
        drugB,
        risk,
        level
    ])


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

result_df = pd.DataFrame(
    results,
    columns=[
        "patient_id",
        "drugA",
        "drugB",
        "risk_score",
        "risk_level"
    ]
)

result_df.to_csv("patient/patient_ddi_predictions.csv", index=False)

print("Prediction completed.")