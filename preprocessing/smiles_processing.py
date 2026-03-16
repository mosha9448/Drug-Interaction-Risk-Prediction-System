"""
smiles_processing.py

Process drug SMILES strings and generate molecular features
using RDKit Morgan fingerprints.

Output:
- drug_mols: RDKit molecule objects
- drug_features: Morgan fingerprint vectors (1024 features)
"""

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
import numpy as np


# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')


def smiles_to_mol(smiles):
    """
    Convert SMILES string to RDKit molecule
    """
    if smiles is None or smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    return mol


def mol_to_fingerprint(mol, radius=2, n_bits=1024):
    """
    Convert RDKit molecule to Morgan fingerprint vector
    """

    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits
    )

    return np.array(fingerprint, dtype=np.float32)


def process_smiles(df):
    """
    Extract unique drugs from dataframe and convert SMILES
    to molecular fingerprints.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns:
        drugA, drugB, smilesA, smilesB

    Returns
    -------
    drug_mols : dict
        drug_name -> RDKit molecule

    drug_features : dict
        drug_name -> fingerprint vector
    """

    print("Extracting unique drugs...")

    drug_smiles = {}

    # Collect unique drugs and their SMILES
    for _, row in df.iterrows():

        drugA = row["drugA"]
        drugB = row["drugB"]

        smilesA = row["smilesA"]
        smilesB = row["smilesB"]

        if drugA not in drug_smiles:
            drug_smiles[drugA] = smilesA

        if drugB not in drug_smiles:
            drug_smiles[drugB] = smilesB

    print("Total unique drugs:", len(drug_smiles))

    print("Processing SMILES structures...")

    drug_mols = {}
    drug_features = {}

    for drug, smiles in drug_smiles.items():

        mol = smiles_to_mol(smiles)

        drug_mols[drug] = mol

        fingerprint = mol_to_fingerprint(mol)

        drug_features[drug] = fingerprint

    print("SMILES processing finished")
    print("Generated molecular fingerprints for all drugs")

    return drug_mols, drug_features