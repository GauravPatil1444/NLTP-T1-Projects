import streamlit as st
import os
import pickle
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
st.set_page_config(page_title="Q&A System in NLP", layout="wide")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'knowledge_base_type' not in st.session_state:
    st.session_state.knowledge_base_type = None

def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    documents = PyPDFLoader(file_path=pdf_path).load()
    split_documents = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, separator="\n").split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_documents, embeddings)
    os.unlink(pdf_path)
    return vectorstore

def save_vectorstore(vectorstore, file_name):
    pkl_filename = f"{os.path.splitext(file_name)[0]}.pkl"
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
    return " ".join([doc.page_content for doc in relevant_docs])

st.title("📚 Document Q&A System")
with st.sidebar:
    st.header("Knowledge Base Settings")
    knowledge_base_type = st.radio("Select Knowledge Base Type", ["PDF Document", "Pre-trained Model (PKL)"])

    if knowledge_base_type == "PDF Document":
        pdf_file = st.file_uploader("Upload PDF Document", type=['pdf'])
        if pdf_file:
            st.session_state.vectorstore = process_pdf(pdf_file)
            st.session_state.knowledge_base_type = "pdf"
            saved_file = save_vectorstore(st.session_state.vectorstore, pdf_file.name)
            st.success(f"PDF processed and saved as '{saved_file}'")
    else:
        pkl_file = st.file_uploader("Upload PKL Model", type=['pkl'])
        if pkl_file:
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