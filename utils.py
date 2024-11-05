import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer
import subprocess

@st.cache_resource
def load_nlp_model():
    # Check if the model is available, and download if not
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Model is missing, so download it
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# import streamlit as st
# import spacy
# from sentence_transformers import SentenceTransformer

# @st.cache_resource
# def load_nlp_model():
#     return spacy.load("en_core_web_sm")

# @st.cache_resource
# def load_transformer_model():
#     return SentenceTransformer('all-MiniLM-L6-v2')
