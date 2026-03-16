import pandas as pd

print("Loading datasets...")

# Load drug dataset
drug_df = pd.read_csv("data/drug_smiles_clean.csv")

# Load patient dataset
patient_df = pd.read_csv("patient/patient_ddi_dataset.csv")

print("Creating drug name → drug_id mapping...")

drug_map = dict(zip(drug_df["name"], drug_df["drug_id"]))

# Map drug names to IDs
patient_df["drugA_id"] = patient_df["drugA"].map(drug_map)
patient_df["drugB_id"] = patient_df["drugB"].map(drug_map)

# Remove rows with missing mapping
patient_df = patient_df.dropna()

print("Total mapped patients:", len(patient_df))

# Save mapped dataset
patient_df.to_csv("patient/patient_drug_mapped.csv", index=False)

print("Saved file → patient/patient_drug_mapped.csv")