import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from preprocessing.patient_processing import process_patient_dataset
from preprocessing.graph_construction import build_ddi_graph
from models.patient_detection_model import PatientDDIModel


print("Starting Patient-Aware DDI Model...")


# --------------------------------------------------
# 1️⃣ Load Drug Interaction Dataset
# --------------------------------------------------

df = pd.read_csv("data/drugbank_ddi_smiles.csv")
df = df.sample(50000)

print("Drug dataset shape:", df.shape)


# --------------------------------------------------
# 2️⃣ Load Patient Datasets
# --------------------------------------------------

patient_data1, columns = process_patient_dataset(
    "patient/clean_patient_dataset.csv"
)

patient_data2, _ = process_patient_dataset(
    "patient/final_mimic_dataset_clean.csv",
    reference_columns=columns
)

patient_features = torch.cat([patient_data1, patient_data2], dim=0)

print("Patient feature tensor shape:", patient_features.shape)


# --------------------------------------------------
# 3️⃣ Build Drug Interaction Graph
# --------------------------------------------------

edge_index, drug_to_idx = build_ddi_graph(df)

num_nodes = len(drug_to_idx)

print("Number of drugs:", num_nodes)


# --------------------------------------------------
# 4️⃣ Node Feature Matrix
# --------------------------------------------------

x = torch.randn(num_nodes, 64)


# --------------------------------------------------
# 5️⃣ Example Drug Pair
# --------------------------------------------------

drugA_idx = torch.tensor([0,1,2,3])
drugB_idx = torch.tensor([1,2,3,4])

labels = torch.tensor([
    [1.0],
    [0.0],
    [1.0],
    [0.0]
])

patient_batch = patient_features[:4]


# --------------------------------------------------
# 6️⃣ Initialize Model
# --------------------------------------------------

model = PatientDDIModel(
    num_nodes,
    patient_dim=patient_features.shape[1]
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --------------------------------------------------
# 7️⃣ Training Loop
# --------------------------------------------------

epochs = 10

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    prediction, ssl = model(
        x,
        edge_index,
        drugA_idx,
        drugB_idx,
        patient_batch
    )

    loss = criterion(prediction, labels) + ssl

    loss.backward()
    optimizer.step()

    print("Epoch", epoch+1, "Loss:", loss.item())


print("Patient-aware training complete")


# --------------------------------------------------
# 8️⃣ Save Model
# --------------------------------------------------

torch.save(model.state_dict(), "models/patient_ddi_model.pth")

print("Patient-aware model saved successfully")