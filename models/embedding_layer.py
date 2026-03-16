import torch.nn as nn


class DrugEmbeddingLayer(nn.Module):

    def __init__(self, input_dim, embedding_dim):

        super(DrugEmbeddingLayer, self).__init__()

        self.linear = nn.Linear(input_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.linear(x)
        x = self.relu(x)

        return x