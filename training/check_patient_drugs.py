import pandas as pd

print("Loading datasets...")

drug_df = pd.read_csv("data/drug_smiles_clean.csv")
patient_df = pd.read_csv("patient/patient_ddi_dataset.csv")

drug_names = set(drug_df["name"].str.lower())

missing = []

for d in patient_df["drugA"]:
    if d.lower() not in drug_names:
        missing.append(d)

for d in patient_df["drugB"]:
    if d.lower() not in drug_names:
        missing.append(d)

missing = set(missing)

print("Missing drugs:", len(missing))
print(missing)