import torch
import torch.nn as nn

from models.mkmgcn_model import MKMGCN
from models.fusion_layer import PatientFusionLayer
from models.detection_layer import DetectionLayer


class PatientDDIModel(nn.Module):

    def __init__(self, num_nodes, patient_dim):

        super(PatientDDIModel, self).__init__()

        # Drug graph encoder
        self.gnn = MKMGCN(num_nodes)

        # Patient fusion layer
        self.fusion = PatientFusionLayer(
            drug_dim=64,
            patient_dim=patient_dim
        )

        # Detection layer
        self.detector = DetectionLayer(128)

    def forward(self, x, edge_index, drugA_idx, drugB_idx, patient_features):

        # Generate drug embeddings
        embeddings, ssl_loss = self.gnn(x, edge_index)

        drugA_emb = embeddings[drugA_idx]
        drugB_emb = embeddings[drugB_idx]

        # Patient-aware fusion
        patient_rep = self.fusion(
            drugA_emb,
            drugB_emb,
            patient_features
        )

        # Predict interaction risk
        prediction = self.detector(patient_rep)

        return prediction, ssl_loss