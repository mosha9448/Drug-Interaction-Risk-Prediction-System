import torch


def build_ddi_graph(df):
    """
    Build Drug-Drug Interaction Graph
    """

    print("Building DDI graph...")

    # collect unique drugs
    drugs = list(set(df["drugA"]).union(set(df["drugB"])))

    # map drug → index
    drug_to_idx = {drug: idx for idx, drug in enumerate(drugs)}

    print("Total nodes:", len(drugs))

    edges = []

    for _, row in df.iterrows():
        a = drug_to_idx[row["drugA"]]
        b = drug_to_idx[row["drugB"]]

        edges.append([a, b])

    edge_index = torch.tensor(edges).t().contiguous()

    print("Total edges:", edge_index.shape[1])

    return edge_index, drug_to_idx