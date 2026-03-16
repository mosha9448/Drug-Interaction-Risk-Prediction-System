
import torch
import torch.nn as nn


class DetectionLayer(nn.Module):

    def __init__(self, hidden_dim, patient_dim=3):

        super().__init__()

        # Drug pair embedding size
        pair_dim = hidden_dim * 2

        # Total input = drug embeddings + patient features
        input_dim = pair_dim + patient_dim

        self.fc_pair = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )


    def forward(self, embA, embB, patient_features):

        # Combine drug embeddings
        pair = torch.cat([embA, embB], dim=1)

        # Add patient features
        x = torch.cat([pair, patient_features], dim=1)

        return self.fc_pair(x)
