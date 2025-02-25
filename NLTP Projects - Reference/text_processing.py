import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import base64

st.set_page_config(
    page_title="Text Preprocessing Tool",
    page_icon="📝",
    layout="wide"
)

@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab','stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        nltk.download(resource)

download_nltk_resources()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return filtered_tokens, stemmed, lemmatized

def get_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Processed CSV</a>'
    return href

st.title("📝 Text Preprocessing steps in NLP")
st.write("Upload your CSV file and process text data using NLTK")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file)
            
        text_column = st.selectbox(
            "Select the text column to process",
            df.columns.tolist()
        )
        
        if st.button("Process Text"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_results = []
            total_rows = len(df)
            
            for idx, text in enumerate(df[text_column]):
                filtered_tokens, stemmed, lemmatized = preprocess_text(text)
                processed_results.append({
                    'filtered_tokens': filtered_tokens,
                    'stemmed': stemmed,
                    'lemmatized': lemmatized
                })
                
                progress = (idx + 1) / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Processing row {idx + 1} of {total_rows}")
                
            df['filtered_tokens'] = [result['filtered_tokens'] for result in processed_results]
            df['stemmed'] = [result['stemmed'] for result in processed_results]
            df['lemmatized'] = [result['lemmatized'] for result in processed_results]
            
            st.subheader("Preview of Processed Data")
            
            tab = st.tabs(["Full Dataset"])
            
            with tab[0]:
                st.dataframe(df)
            
            st.markdown("### Download Processed Data")
            st.markdown(get_download_link(df, "processed_data.csv"), unsafe_allow_html=True)
            
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

with st.sidebar:
    st.header("About")
    st.write("""
    This tool helps you preprocess text data using NLTK. It performs:
    - Tokenization
    - Stop Word Removal
    - Stemming
    - Lemmatization
    """)
    
    st.header("Instructions")
    st.write("""
    1. Upload a CSV file
    2. Select the text column to process
    3. Click 'Process Text'
    4. View results and download processed data
    """)