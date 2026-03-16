import torch
import torch.nn as nn


class PatientFusionLayer(nn.Module):

    def __init__(self, drug_dim=64, patient_dim=10):

        super(PatientFusionLayer, self).__init__()

        # Drug pair fusion
        self.drug_fusion = nn.Sequential(
            nn.Linear(drug_dim * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )

        # Patient context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(128 + patient_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )

    def forward(self, drugA_emb, drugB_emb, patient_features):

        # Drug pair fusion
        drug_pair = torch.cat([drugA_emb, drugB_emb], dim=1)

        drug_pair_rep = self.drug_fusion(drug_pair)

        # Context-aware fusion
        combined = torch.cat([drug_pair_rep, patient_features], dim=1)

        patient_specific_rep = self.context_fusion(combined)

        return patient_specific_rep