import torch
from torch.utils.data import DataLoader, TensorDataset


def load_training_data(df, drug_to_idx):

    drugA_idx = []
    drugB_idx = []
    labels = []

    for _, row in df.iterrows():

        if row["drugA"] not in drug_to_idx:
            continue

        if row["drugB"] not in drug_to_idx:
            continue

        drugA_idx.append(drug_to_idx[row["drugA"]])
        drugB_idx.append(drug_to_idx[row["drugB"]])
        labels.append(row["interaction"])

    drugA_idx = torch.tensor(drugA_idx)
    drugB_idx = torch.tensor(drugB_idx)
    labels = torch.tensor(labels).float().view(-1, 1)

    dataset = TensorDataset(drugA_idx, drugB_idx, labels)

    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    return loader