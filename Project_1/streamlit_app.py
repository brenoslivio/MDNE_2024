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

def extract_embeddings(sequence, tokenizer, model, device):
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence)))]

    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    embedding = embedding.cpu().numpy()

    seq_len = (attention_mask[0] == 1).sum()
    seq_emd = embedding[0][:seq_len-1]

    avg_pool = seq_emd.mean(axis=0)

    dict_embeddings = {f"embedding_{i}": embed for i, embed in enumerate(avg_pool)}

    return dict_embeddings

def seqs_embeddings(df_seqs):
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model = model.eval()

    df_seqs_embeddings = df_seqs.with_columns(pl.col("sequence").map_elements(lambda x: extract_embeddings(x, tokenizer, model, device)).alias("embeddings")).unnest("embeddings")
    
    return df_seqs_embeddings

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

def runUI():
    st.set_page_config(page_title="MDNE 2024 - Projeto 1", layout="wide")
    
    st.title("Projeto 1: Mineração de Textos - Classificação de Peptídeos Anti Câncer")

    # Initialize text box with empty string
    fasta_input = st.text_area("Enter protein sequences in FASTA format", key="job_input", height=200)
    
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Submit"):
            if fasta_input:
                df_seqs = parse_fasta(fasta_input)
                st.write("Parsed Sequences:")
                st.write(df_seqs)

                with st.spinner("Computing embeddings..."):
                    df_seqs_embeddings = seqs_embeddings(df_seqs)

                st.write("Computed Embeddings:")
                st.write(df_seqs_embeddings)
            else:
                st.error("Please input sequences in FASTA format.")

    with col2:
        example = st.button("Example", on_click=get_example_sequences)




if __name__ == "__main__":
    runUI()