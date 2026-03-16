import torch


def compute_magnetic_laplacian(edge_index, num_nodes):
    """
    Compute direction-aware magnetic Laplacian matrix
    """

    adj = torch.zeros((num_nodes, num_nodes))

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        adj[src, dst] = 1

    degree = torch.diag(adj.sum(dim=1))

    laplacian = degree - adj

    return laplacian