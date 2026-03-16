
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

from models.mkmgcn_model import MKMGCN
from preprocessing.graph_construction import build_ddi_graph
from preprocessing.smiles_processing import process_smiles


print("Starting LIME explanation...")

device = torch.device("cpu")

# --------------------------------------------------
# Load datasets
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

print("Processing SMILES...")

smiles_df = pd.DataFrame({
    "drugA": df["drugA"],
    "drugB": df["drugB"],
    "smilesA": df["smilesA"],
    "smilesB": df["smilesB"]
})

drug_mols, drug_features = process_smiles(smiles_df)


# --------------------------------------------------
# Build DDI graph
# --------------------------------------------------

print("Building DDI graph...")

edge_index, drug_to_idx = build_ddi_graph(df)

num_nodes = len(drug_to_idx)


# --------------------------------------------------
# Feature matrix
# --------------------------------------------------

feature_dim = len(next(iter(drug_features.values())))
feature_matrix = np.zeros((num_nodes, feature_dim))

for drug, idx in drug_to_idx.items():
    feature_matrix[idx] = drug_features[drug]

x = torch.tensor(feature_matrix).float()
edge_index = edge_index


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

print("Loading trained model...")

model = MKMGCN(num_nodes)
model.load_state_dict(torch.load("models/ddi_model.pth", map_location=device))
model.eval()


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

with torch.no_grad():
    embeddings, _ = model(x, edge_index)

hidden_dim = embeddings.shape[1]

print("Embedding dimension:", hidden_dim)


# --------------------------------------------------
# Create sample patient input
# --------------------------------------------------

drugA_id = list(drug_to_idx.keys())[0]
drugB_id = list(drug_to_idx.keys())[1]

idxA = drug_to_idx[drugA_id]
idxB = drug_to_idx[drugB_id]

embA = embeddings[idxA]
embB = embeddings[idxB]

age = torch.tensor([0.65])
gender = torch.tensor([1.0])
disease = torch.tensor([2.0])

patient_features = torch.cat([age, gender, disease])

input_vector = torch.cat([embA, embB, patient_features])


# --------------------------------------------------
# Prediction function for LIME
# --------------------------------------------------

def model_predict(data):

    data = torch.tensor(data).float()

    embA = data[:, :hidden_dim]
    embB = data[:, hidden_dim:hidden_dim*2]
    patient = data[:, hidden_dim*2:]

    with torch.no_grad():
        pred = model.detector(embA, embB, patient)

    return np.concatenate([1 - pred.numpy(), pred.numpy()], axis=1)


# --------------------------------------------------
# Create synthetic dataset for LIME
# --------------------------------------------------

samples = 200

random_data = np.random.normal(
    loc=input_vector.numpy(),
    scale=0.1,
    size=(samples, len(input_vector))
)


# --------------------------------------------------
# Feature names
# --------------------------------------------------

feature_names = (
    ["DrugA_emb"] * hidden_dim +
    ["DrugB_emb"] * hidden_dim +
    ["Age", "Gender", "Disease"]
)


# --------------------------------------------------
# LIME Explainer
# --------------------------------------------------

explainer = LimeTabularExplainer(
    training_data=random_data,
    feature_names=feature_names,
    class_names=["Low Risk", "High Risk"],
    mode="classification"
)


# --------------------------------------------------
# Explain prediction
# --------------------------------------------------

exp = explainer.explain_instance(
    data_row=input_vector.numpy(),
    predict_fn=model_predict,
    num_features=10
)


# --------------------------------------------------
# Print explanation
# --------------------------------------------------

print("\nTop features influencing prediction:\n")

for feature, weight in exp.as_list():
    print(f"{feature} : {weight:.4f}")


# --------------------------------------------------
# Show explanation plot
# --------------------------------------------------

fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.show()