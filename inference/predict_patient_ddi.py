import pandas as pd
import torch

from models.mkmgcn_model import MKMGCN   # import model architecture

print("Loading patient dataset...")

df = pd.read_csv("patient/patient_drug_mapped.csv")

print("Loading trained model...")

# initialize model
model = MKMGCN()

# load trained weights
model.load_state_dict(torch.load("models/patient_ddi_model.pth", map_location="cpu"))

model.eval()

results = []

for _, row in df.iterrows():

    drugA = int(row["drugA_id"])
    drugB = int(row["drugB_id"])

    x = torch.tensor([[drugA, drugB]], dtype=torch.float32)

    with torch.no_grad():
        risk = model(x).item()

    if risk > 0.7:
        label = "High"
    elif risk > 0.4:
        label = "Medium"
    else:
        label = "Low"

    results.append([
        row["patient_id"],
        row["drugA"],
        row["drugB"],
        risk,
        label
    ])

result_df = pd.DataFrame(
    results,
    columns=[
        "patient_id",
        "drugA",
        "drugB",
        "risk_score",
        "risk_level"
    ]
)

result_df.to_csv("patient/patient_ddi_predictions.csv", index=False)

print("Prediction finished")
print("Saved → patient/patient_ddi_predictions.csv")