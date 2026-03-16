import sys
import os

# add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd

from models.mkmgcn_model import MKMGCN
from preprocessing.graph_construction import build_ddi_graph
from preprocessing.smiles_processing import process_smiles
from training.dataset_loader import load_training_data

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print("Loading dataset...")

df = pd.read_csv("data/drug_interactions.csv")

df = df.rename(columns={
    "drug_id": "drugA",
    "interacting_drug_id": "drugB"
})

df["interaction"] = 1


print("Building graph...")

edge_index, drug_to_idx = build_ddi_graph(df)


print("Preparing loader...")

loader = load_training_data(df, drug_to_idx)


print("Loading model...")

num_nodes = len(drug_to_idx)

model = MKMGCN(num_nodes)

model.load_state_dict(torch.load("models/ddi_model.pth"))

model.eval()


all_preds = []
all_labels = []


for drugA, drugB, label in loader:

    embA = torch.randn(len(drugA), 128)
    embB = torch.randn(len(drugB), 128)

    pred = model.detector(embA, embB)

    all_preds.extend(pred.detach().numpy())
    all_labels.extend(label.numpy())


preds = (torch.tensor(all_preds) > 0.5).int()


acc = accuracy_score(all_labels, preds)
prec = precision_score(all_labels, preds)
rec = recall_score(all_labels, preds)
f1 = f1_score(all_labels, preds)


print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)