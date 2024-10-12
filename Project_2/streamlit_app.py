import streamlit as st
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import joblib
import urllib.request

def extract_embeddings(img):
    model = models.resnet18(pretrained=True)

    layer = model._modules.get('avgpool')
    model.eval()

    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    embedding = torch.zeros(512)

    def capture_embedding(m, i, o):
        embedding.copy_(o.data.reshape(o.data.size(1)))

    h = layer.register_forward_hook(capture_embedding)
    model(img)

    h.remove()

    embeddings_feat = {}

    for i, feature in enumerate(np.array(embedding)):
      embeddings_feat[f"embedding_{i}"] = feature

    return pl.DataFrame(embeddings_feat)

def predict_image(df_image_embedding):
    X = df_image_embedding.to_numpy()

    response = urllib.request.urlopen("https://github.com/brenoslivio/MDNE_2024/raw/refs/heads/main/Project_2/best_model_embed.pkl")
    best_model = joblib.load(response)

    y_pred = pl.DataFrame(best_model.predict(X))

    y_pred.columns = ["prediction"]

    return y_pred

def runUI():
    st.set_page_config(page_title="MDNE 2024 - Projeto 1", layout="wide")
    
    st.title("Projeto 2: Mineração de Imagens - Classificação de Abelhas e Vespas")

    uploaded_file = st.file_uploader("Coloque uma imagem de abelha ou vespa para predição", type=["jpg", "png", "jpeg"])

    img_container = st.container()

    submit = st.button("Submeter")

    with img_container:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem para submissão.", width=400)

    if submit:
        if uploaded_file:
            
            with st.spinner("Calculando Embeddings..."):
                df_image_embedding = extract_embeddings(image)
            
            st.markdown("**Representação da imagem em Embeddings:**")
            st.dataframe(df_image_embedding, use_container_width=True)

            y_pred = predict_image(df_image_embedding)

            st.markdown("**Resultados da predição:**")
            
            st.markdown(f"A imagem se trata de uma **{y_pred['prediction'].item()}**")
            # st.dataframe(pl.concat([df_seqs, y_pred], how="horizontal"), use_container_width=True)
        else:
            st.error("Coloque a imagem no formato jpg, png ou jpeg.")

if __name__ == "__main__":
    runUI()