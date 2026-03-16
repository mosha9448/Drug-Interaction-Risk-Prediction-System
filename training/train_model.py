
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset

from preprocessing.smiles_processing import process_smiles
from preprocessing.graph_construction import build_ddi_graph
from models.mkmgcn_model import MKMGCN


def main():

    print("Starting DDI Training...")

    torch.set_num_threads(16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    ddi = pd.read_csv("data/drug_interactions.csv")
    smiles = pd.read_csv("data/drug_smiles_clean.csv")

    ddi = ddi.rename(columns={
        "drug_id": "drugA",
        "interacting_drug_id": "drugB"
    })

    print("Merging SMILES...")

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

    df = df.dropna(subset=["smilesA","smilesB"])
    df["interaction"] = 1

    print("Dataset shape:", df.shape)
    print("Unique drugs:", len(set(df["drugA"]).union(set(df["drugB"]))))

    # --------------------------------------------------
    # Negative sampling
    # --------------------------------------------------

    neg_cache = os.path.join(cache_dir, "negative_samples.csv")

    if os.path.exists(neg_cache):

        print("Loading cached negative samples...")
        neg_df = pd.read_csv(neg_cache)

    else:

        print("Generating negative samples...")

        n = len(df)
        drug_ids = np.array(list(set(df["drugA"]).union(set(df["drugB"]))))

        drugA_neg = np.random.choice(drug_ids, n)
        drugB_neg = np.random.choice(drug_ids, n)

        neg_df = pd.DataFrame({
            "drugA": drugA_neg,
            "drugB": drugB_neg,
            "interaction": 0
        })

        neg_df = neg_df[neg_df["drugA"] != neg_df["drugB"]]

        neg_df.to_csv(neg_cache, index=False)

    df_train = pd.concat([
        df[["drugA","drugB","interaction"]],
        neg_df
    ])

    print("Total training pairs:", len(df_train))

    # --------------------------------------------------
    # SMILES Processing
    # --------------------------------------------------

    smiles_cache = os.path.join(cache_dir,"drug_features.pkl")

    smiles_df = pd.DataFrame({
        "drugA": df["drugA"],
        "drugB": df["drugB"],
        "smilesA": df["smilesA"],
        "smilesB": df["smilesB"]
    })

    if os.path.exists(smiles_cache):

        print("Loading cached SMILES features...")
        with open(smiles_cache,"rb") as f:
            drug_features = pickle.load(f)

    else:

        print("Processing SMILES...")

        drug_mols, drug_features = process_smiles(smiles_df)

        with open(smiles_cache,"wb") as f:
            pickle.dump(drug_features,f)

    # --------------------------------------------------
    # Graph construction
    # --------------------------------------------------

    graph_cache = os.path.join(cache_dir,"ddi_graph.pt")

    if os.path.exists(graph_cache):

        print("Loading cached graph...")
        edge_index, drug_to_idx = torch.load(graph_cache)

    else:

        print("Building DDI graph...")
        edge_index, drug_to_idx = build_ddi_graph(df)

        torch.save((edge_index,drug_to_idx),graph_cache)

    num_nodes = len(drug_to_idx)
    print("Number of drugs:", num_nodes)

    # --------------------------------------------------
    # Feature matrix
    # --------------------------------------------------

    feature_dim = len(next(iter(drug_features.values())))
    feature_matrix = np.zeros((num_nodes,feature_dim))

    for drug,idx in drug_to_idx.items():
        feature_matrix[idx] = drug_features[drug]

    x = torch.tensor(feature_matrix).float().to(device)

    # --------------------------------------------------
    # Training dataset
    # --------------------------------------------------

    drugA_idx = df_train["drugA"].map(drug_to_idx).values
    drugB_idx = df_train["drugB"].map(drug_to_idx).values
    labels = df_train["interaction"].values

    dataset = TensorDataset(
        torch.tensor(drugA_idx),
        torch.tensor(drugB_idx),
        torch.tensor(labels).float()
    )

    loader = DataLoader(
        dataset,
        batch_size=16384,
        shuffle=True
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("Initializing MKMGCN model...")

    model = MKMGCN(num_nodes).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    epochs = 10

    print("Training started...")

    edge_index = edge_index.to(device)

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        total_samples = 0

        embeddings, ssl_loss = model(x,edge_index)
        embeddings = embeddings.detach()

        for drugA,drugB,label in loader:

            drugA = drugA.to(device)
            drugB = drugB.to(device)
            label = label.to(device).unsqueeze(1)

            optimizer.zero_grad()

            embA = embeddings[drugA]
            embB = embeddings[drugB]

            batch_size = drugA.size(0)

            # --------------------------------------------------
            # Temporary patient features
            # --------------------------------------------------

            age = torch.rand(batch_size,1).to(device)
            gender = torch.randint(0,2,(batch_size,1)).float().to(device)
            disease = torch.randint(0,5,(batch_size,1)).float().to(device)

            patient_features = torch.cat([age,gender,disease],dim=1)

            # --------------------------------------------------

            pred = model.detector(embA,embB,patient_features)

            loss = criterion(pred,label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()*batch_size
            total_samples += batch_size

        avg_loss = total_loss/total_samples

        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

    print("Training complete")

    os.makedirs("models",exist_ok=True)

    torch.save(model.state_dict(),"models/ddi_model.pth")

    print("Model saved successfully")


if __name__ == "__main__":
    main()
