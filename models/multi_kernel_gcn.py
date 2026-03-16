import torch.nn as nn
from torch_geometric.nn import GCNConv


class MultiKernelGCN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(MultiKernelGCN, self).__init__()

        # Low-frequency kernel
        self.kernel_low = GCNConv(input_dim, hidden_dim)

        # High-frequency kernel
        self.kernel_high = GCNConv(input_dim, hidden_dim)

    def forward(self, x, edge_index):

        low_features = self.kernel_low(x, edge_index)

        high_features = self.kernel_high(x, edge_index)

        return low_features, high_features