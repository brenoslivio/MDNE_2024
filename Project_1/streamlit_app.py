import streamlit as st
import requests, io
import polars as pl
import numpy as np
from sklearn.svm import SVC
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from transformers import T5EncoderModel, T5Tokenizer
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, matthews_corrcoef, recall_score, confusion_matrix
import torch
import re
import joblib
import urllib.request

def dict_kmer(seq, k):
    chars = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    
    counts = {''.join(comb): 0 for comb in product(chars, repeat= k)}
    L = len(seq)
    for i in range(L - k + 1):
        counts[seq[i:i+k]] += 1

    return counts

def seqs_bow(df_seqs):
    df_seqs_bow = df_seqs.with_columns(pl.col("sequence").map_elements(lambda x: dict_kmer(x, 1)).alias("aac")).unnest("aac")

    return df_seqs_bow

def parse_fasta(fasta_text):
    """Parse a FASTA format text and return a polars DataFrame with columns 'header' and 'sequence'."""
    sequences = []
    headers = []
    sequence = []
    
    for line in fasta_text.strip().splitlines():
        if line.startswith(">"):
            if sequence:
                sequences.append("".join(sequence))
                sequence = []
            headers.append(line[1:].strip())  # Remove '>' and strip spaces
        else:
            sequence.append(line.strip())
    
    # Add the last sequence
    if sequence:
        sequences.append("".join(sequence))
    
    # Create polars DataFrame
    df = pl.DataFrame({
        "header": headers,
        "sequence": sequences
    })
    
    return df


def get_example_sequences():
    """Returns example protein sequences in FASTA format."""
    st.session_state["job_input"] = """>sequence1
MKTIIALSYIFCLVFAD
>sequence2
GATTACAACGGGTAAGTCTG
>sequence3
GGTAACGGGTTTACGCTTAT
"""

def predict_sequences(df_seqs):
    X = df_seqs.select(pl.nth(range(2, len(df_seqs.columns)))).to_numpy()

    response = urllib.request.urlopen("https://github.com/brenoslivio/MDNE_2024/raw/refs/heads/main/Project_1/best_model.pkl")
    best_model = joblib.load(response)

    y_pred = pl.DataFrame(best_model.predict(X))

    y_pred.columns = ["prediction"]

    return y_pred

def runUI():
    st.set_page_config(page_title="MDNE 2024 - Projeto 1", layout="wide")
    
    st.title("Projeto 1: Mineração de Textos - Classificação de Peptídeos Anti Câncer")

    # Initialize text box with empty string
    fasta_input = st.text_area("Coloque as sequências de proteínas em formato FASTA", key="job_input", height=200)
    
    col1, col2 = st.columns(2)

    with col1:
        submit = st.button("Submeter")

    with col2:
        example = st.button("Exemplo", on_click=get_example_sequences)

    if submit:
        if fasta_input:
            df_seqs = parse_fasta(fasta_input)
            st.markdown("**Sequências lidas:**")
            st.dataframe(df_seqs, use_container_width=True)

            with st.spinner("Calculando Bag of Words..."):
                df_seqs_bow = seqs_bow(df_seqs)

            st.markdown("**Representação em Bag of Words:**")
            st.dataframe(df_seqs_bow, use_container_width=True)

            y_pred = predict_sequences(df_seqs_bow)

            st.markdown("**Resultados da predição:**")
            st.dataframe(pl.concat([df_seqs, y_pred],how="horizontal"), use_container_width=True)
        else:
            st.error("Coloque as sequências no formato FASTA.")

if __name__ == "__main__":
    runUI()