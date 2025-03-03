import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import base64

@st.cache_resource
def initialize_nltk_resources():
    resources_to_download = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources_to_download:
        nltk.download(resource)

initialize_nltk_resources()

nlp_stop_words = set(stopwords.words('english'))
stemmer_tool = PorterStemmer()
lemmatizer_tool = WordNetLemmatizer()

def process_text(input_text):
    tokenized_words = word_tokenize(str(input_text).lower())
    valid_tokens = [token for token in tokenized_words if token.isalpha() and token not in nlp_stop_words]
    stemmed_words = [stemmer_tool.stem(token) for token in valid_tokens]
    lemmatized_words = [lemmatizer_tool.lemmatize(token) for token in valid_tokens]
    return valid_tokens, stemmed_words, lemmatized_words

def generate_download_link(dataframe, filename):
    csv_data = dataframe.to_csv(index=False)
    encoded_csv = base64.b64encode(csv_data.encode()).decode()
    download_link = f'<a href="data:file/csv;base64,{encoded_csv}" download="{filename}">Download Processed CSV</a>'
    return download_link

st.title("🖋 NLP Preprocessing Steps")
st.write("Upload a CSV file to preprocess text data using NLTK.")

file_upload = st.file_uploader("Select a CSV file", type="csv")

if file_upload is not None:
    try:
        with st.spinner("Reading file..."):
            input_dataframe = pd.read_csv(file_upload)

        column_to_process = st.selectbox(
            "Choose the column for text preprocessing",
            input_dataframe.columns.tolist()
        )

        if st.button("Start Processing"):
            progress_bar = st.progress(0)
            processing_status = st.empty()

            processed_data = []
            total_records = len(input_dataframe)

            for index, row_text in enumerate(input_dataframe[column_to_process]):
                tokens, stems, lemmas = process_text(row_text)
                processed_data.append({
                    'filtered_tokens': tokens,
                    'stemmed': stems,
                    'lemmatized': lemmas
                })

                progress = (index + 1) / total_records
                progress_bar.progress(progress)
                processing_status.text(f"Processing row {index + 1} of {total_records}")

            input_dataframe['filtered_tokens'] = [item['filtered_tokens'] for item in processed_data]
            input_dataframe['stemmed'] = [item['stemmed'] for item in processed_data]
            input_dataframe['lemmatized'] = [item['lemmatized'] for item in processed_data]

            st.subheader("Processed Data Preview")

            tab = st.tabs(["Dataset View"])

            with tab[0]:
                st.dataframe(input_dataframe)

    except Exception as error:
        st.error(f"An error occurred: {str(error)}")
