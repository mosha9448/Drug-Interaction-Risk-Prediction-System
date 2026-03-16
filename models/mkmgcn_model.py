import torch
import torch.nn as nn

from models.magnetic_laplacian import compute_magnetic_laplacian
from models.multi_kernel_gcn import MultiKernelGCN
from models.feature_aggregation import aggregate_features
from models.embedding_layer import DrugEmbeddingLayer
from models.ssl_module import ssl_loss
from models.detection_layer import DetectionLayer


class MKMGCN(nn.Module):

    def __init__(self, num_nodes, input_dim=1024, hidden_dim=64):

        super(MKMGCN, self).__init__()

        self.num_nodes = num_nodes

        # Multi-kernel GCN
        self.gcn = MultiKernelGCN(input_dim, hidden_dim)

        # Embedding layer
        self.embedding_layer = DrugEmbeddingLayer(hidden_dim * 2, hidden_dim)

        # Interaction detection layer
        self.detector = DetectionLayer(hidden_dim)

    def forward(self, x, edge_index, drugA_idx=None, drugB_idx=None):

        # Magnetic Laplacian
        laplacian = compute_magnetic_laplacian(edge_index, self.num_nodes)

        # Multi-kernel graph convolution
        low_feat, high_feat = self.gcn(x, edge_index)

        # Feature aggregation
        fused = aggregate_features(low_feat, high_feat)

        # Drug embeddings
        embeddings = self.embedding_layer(fused)

        # Self-supervised loss
        ssl = ssl_loss(embeddings)

        if drugA_idx is None:
            return embeddings, ssl

        embA = embeddings[drugA_idx]
        embB = embeddings[drugB_idx]

        prediction = self.detector(embA, embB)

        return prediction, ssl