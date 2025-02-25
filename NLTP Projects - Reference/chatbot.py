import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# import requests
import pickle
import tempfile

load_dotenv()
st.set_page_config(page_title="Q&A System in NLP", layout="wide")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'knowledge_base_type' not in st.session_state:
    st.session_state.knowledge_base_type = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None

def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, separator="\n")
    split_documents = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_documents, embeddings)
    
    os.unlink(pdf_path)  
    return vectorstore

def save_vectorstore(vectorstore, pdf_name):
    base_name = os.path.splitext(pdf_name)[0]
    pkl_filename = f"{base_name}.pkl"
    with open(pkl_filename, "wb") as f:
        pickle.dump(vectorstore, f)
    return pkl_filename

def load_vectorstore(pkl_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        tmp_file.write(pkl_file.getvalue())
        pkl_path = tmp_file.name

    with open(pkl_path, "rb") as f:
        vectorstore = pickle.load(f)
    
    os.unlink(pkl_path) 
    return vectorstore

def get_qa_response(vectorstore, question):
    relevant_docs = vectorstore.similarity_search(question)
    context = " ".join([doc.page_content for doc in relevant_docs])
    
    return context

st.title("📚 Document Q&A System")

with st.sidebar:
    st.header("Knowledge Base Settings")
    knowledge_base_type = st.radio(
        "Select Knowledge Base Type",
        ["PDF Document", "Pre-trained Model (PKL)"]
    )

    if knowledge_base_type == "PDF Document":
        pdf_file = st.file_uploader("Upload PDF Document", type=['pdf'])
        if pdf_file and st.session_state.knowledge_base_type != "pdf":
            st.session_state.current_file_name = pdf_file.name
            with st.spinner("Processing PDF..."):
                st.session_state.vectorstore = process_pdf(pdf_file)
                st.session_state.knowledge_base_type = "pdf"
                
                saved_file = save_vectorstore(st.session_state.vectorstore, st.session_state.current_file_name)
                st.success(f"PDF processed and saved as '{saved_file}'")

    else:
        pkl_file = st.file_uploader("Upload PKL Model", type=['pkl'])
        if pkl_file and st.session_state.knowledge_base_type != "pkl":
            with st.spinner("Loading PKL model..."):
                st.session_state.vectorstore = load_vectorstore(pkl_file)
                st.session_state.knowledge_base_type = "pkl"
                st.success("Model loaded successfully!")

st.header("Ask Questions")

if st.session_state.vectorstore is None:
    st.info("Please upload a PDF document or PKL model to start asking questions.")
else:
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Getting answer..."):
            answer = get_qa_response(st.session_state.vectorstore, question)
            st.write("Answer:", answer)