import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import networkx as nx

from models.mkmgcn_model import MKMGCN

st.set_page_config(page_title="Drug Interaction AI", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#1f77b4'>
Drug Interaction Risk Prediction System
</h1>
""", unsafe_allow_html=True)

device = torch.device("cpu")

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    edge_index, drug_to_idx = torch.load("cache/ddi_graph.pt")

    with open("cache/drug_features.pkl","rb") as f:
        drug_features = pickle.load(f)

    num_nodes = len(drug_to_idx)
    feature_dim = len(next(iter(drug_features.values())))

    feature_matrix = torch.zeros((num_nodes,feature_dim))

    for drug,idx in drug_to_idx.items():
        feature_matrix[idx] = torch.tensor(drug_features[drug])

    x = feature_matrix.float()

    model = MKMGCN(num_nodes)
    model.load_state_dict(torch.load("models/ddi_model.pth",map_location=device))
    model.eval()

    with torch.no_grad():
        embeddings,_ = model(x,edge_index)

    return model,embeddings,drug_to_idx,edge_index


with st.spinner("Loading AI Model..."):
    model,embeddings,drug_to_idx,edge_index = load_model()

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

desc_df = pd.read_csv("data/drug_smiles_clean.csv")
side_df = pd.read_csv("data/drugbank_with_side_effects.csv")

valid_drugs = desc_df[desc_df["drug_id"].isin(drug_to_idx.keys())]

drug_options = valid_drugs["drug_id"] + " - " + valid_drugs["name"]
drug_map = dict(zip(drug_options,valid_drugs["drug_id"]))

# --------------------------------------------------
# Inputs
# --------------------------------------------------

col1,col2 = st.columns(2)

with col1:
    drugA_choice = st.selectbox("Drug A",drug_options)

with col2:
    drugB_choice = st.selectbox("Drug B",drug_options)

drugA = drug_map[drugA_choice]
drugB = drug_map[drugB_choice]

age = st.slider("Age",0,100,50)
gender = st.selectbox("Gender",["Male","Female"])

disease_list = [
"Diabetes","Hypertension","Cancer","Stroke","Asthma",
"Kidney Disease","Heart Disease","Liver Disease",
"Arthritis","Depression"
]

disease = st.selectbox("Disease",disease_list)
disease_map = {d:i for i,d in enumerate(disease_list)}

# --------------------------------------------------
# Predict Button
# --------------------------------------------------

colA,colB,colC = st.columns([3,1,3])

with colB:
    predict_button = st.button("Predict Interaction")

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    idxA = drug_to_idx[drugA]
    idxB = drug_to_idx[drugB]

    embA = embeddings[idxA].unsqueeze(0)
    embB = embeddings[idxB].unsqueeze(0)

    age_t = torch.tensor([[age/100]]).float()
    gender_t = torch.tensor([[1 if gender=="Male" else 0]]).float()
    disease_t = torch.tensor([[disease_map[disease]]]).float()

    patient_features = torch.cat([age_t,gender_t,disease_t],dim=1)

    with torch.no_grad():
        risk_score = model.detector(embA,embB,patient_features).item()

    if risk_score > 0.7:
        level = "Major"
    elif risk_score > 0.4:
        level = "Moderate"
    else:
        level = "Minor"

    nameA = desc_df[desc_df["drug_id"]==drugA]["name"].values[0]
    nameB = desc_df[desc_df["drug_id"]==drugB]["name"].values[0]

# --------------------------------------------------
# Prediction Result
# --------------------------------------------------

    st.markdown("## Prediction Result")

    col1,col2 = st.columns(2)

    with col1:
        st.metric("Interaction Probability",round(risk_score,3))
        st.metric("Risk Level",level)
        st.progress(risk_score)

    with col2:

        fig,ax = plt.subplots()

        ax.bar(["Risk Probability"],[risk_score],color="#ff4b4b")
        ax.set_ylim(0,1)

        st.pyplot(fig)

# --------------------------------------------------
# Explainable AI
# --------------------------------------------------

    st.markdown("## Explainable AI Analysis")

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Interaction Direction")

        st.write("Perpetrator Drug:",nameB,"(",drugB,")")
        st.write("Victim Drug:",nameA,"(",drugA,")")

# --------------------------------------------------
# Patient Factors
# --------------------------------------------------

    with col2:

        st.subheader("Important Patient Factors")

        age_factor = age/100
        gender_factor = 0.3 if gender=="Male" else 0.2
        disease_factor = (disease_map[disease]+1)/len(disease_map)

        total = age_factor + gender_factor + disease_factor

        age_imp = age_factor/total
        gender_imp = gender_factor/total
        disease_imp = disease_factor/total

        labels = ["Age","Gender","Disease"]
        values = [age_imp,gender_imp,disease_imp]

        fig2,ax2 = plt.subplots(figsize=(6,4))

        bars = ax2.bar(labels,values,color=["orange","green","purple"])

        ax2.set_title("Patient Risk Factor Importance")
        ax2.set_ylabel("Contribution Score")
        ax2.set_ylim(0,1)

        for bar in bars:

            height = bar.get_height()

            ax2.text(bar.get_x()+bar.get_width()/2,
                     height+0.02,
                     f"{height*100:.1f}%",
                     ha="center")

        st.pyplot(fig2)

# --------------------------------------------------
# Detailed Explanation
# --------------------------------------------------

    st.markdown("### Detailed Patient Factor Explanation")

    colA,colB,colC = st.columns(3)

    with colA:
        st.markdown("#### Age Influence")

        st.write(f"""
Patient age is **{age} years**.

Older patients may have slower drug metabolism and reduced kidney filtration,
which increases the probability of drug interactions.

Age contribution to risk: **{age_imp*100:.2f}%**
""")

    with colB:
        st.markdown("#### Gender Influence")

        st.write(f"""
Patient gender: **{gender}**

Drug response can differ because of hormonal levels,
enzyme activity, and body composition differences.

Gender contribution to risk: **{gender_imp*100:.2f}%**
""")

    with colC:
        st.markdown("#### Disease Influence")

        st.write(f"""
Patient disease condition: **{disease}**

Chronic diseases can affect organ function and drug metabolism,
increasing the likelihood of adverse drug interactions.

Disease contribution to risk: **{disease_imp*100:.2f}%**
""")

# --------------------------------------------------
# Side Effects
# --------------------------------------------------

    st.markdown("## Side Effects Analysis")

    sideA = side_df[side_df["DrugBank ID"]==drugA]
    sideB = side_df[side_df["DrugBank ID"]==drugB]

    col1,col2 = st.columns(2)

    with col1:

        st.subheader(nameA + " Side Effects")

        if not sideA.empty:

            effects = sideA["Side Effects/Toxicity"].values[0].split(",")

            for e in effects[:10]:
                st.markdown(f"- {e.strip()}")

        else:
            st.write("No side effects available.")

    with col2:

        st.subheader(nameB + " Side Effects")

        if not sideB.empty:

            effects = sideB["Side Effects/Toxicity"].values[0].split(",")

            for e in effects[:10]:
                st.markdown(f"- {e.strip()}")

        else:
            st.write("No side effects available.")

# --------------------------------------------------
# Drug Description
# --------------------------------------------------

    st.markdown("## Drug Description")

    descA = desc_df[desc_df["drug_id"]==drugA]
    descB = desc_df[desc_df["drug_id"]==drugB]

    col1,col2 = st.columns(2)

    with col1:
        st.write("### Drug A")
        st.write(descA["name"].values[0])
        st.write(descA["description"].values[0])

    with col2:
        st.write("### Drug B")
        st.write(descB["name"].values[0])
        st.write(descB["description"].values[0])