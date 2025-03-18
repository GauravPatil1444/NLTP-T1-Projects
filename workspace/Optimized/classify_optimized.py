import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spam Classification System", page_icon="📧", layout="wide")

@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_resources()

@st.cache
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in word_tokenize(str(text).lower()) if word.isalpha() and word not in stop_words)

def encode_labels(df, label_column):
    return df[label_column].map({'ham': 0, 'spam': 1})

def train_model(df, text_column, label_column):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[text_column])
    y = encode_labels(df, label_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB().fit(X_train, y_train)
    return model, vectorizer, y_test, model.predict(X_test)

def save_model(model, vectorizer, prefix):
    for obj, name in zip([model, vectorizer], ["model", "vectorizer"]):
        with open(f'{prefix}_{name}.pkl', 'wb') as f:
            pickle.dump(obj, f)

def load_model(prefix):
    try:
        with open(f'{prefix}_model.pkl', 'rb') as f1, open(f'{prefix}_vectorizer.pkl', 'rb') as f2:
            return pickle.load(f1), pickle.load(f2)
    except FileNotFoundError:
        return None, None

def plot_confusion_matrix(y_true, y_pred):
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot()

st.title("📧 Spam Classification System")
tabs = st.tabs(["Train Model", "Classify Email"])

with tabs[0]:
    st.header("Train New Model")
    uploaded_file = st.file_uploader("Upload training data (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        text_column, label_column = st.selectbox("Select text column", df.columns), st.selectbox("Select label column", df.columns)
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                df['processed_text'] = df[text_column].apply(preprocess_text)
                model, vectorizer, y_test, y_pred = train_model(df, 'processed_text', label_column)
                save_model(model, vectorizer, 'spam_classifier')
                st.success("Model trained successfully!")
                st.text(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM']))
                plot_confusion_matrix(y_test, y_pred)

with tabs[1]:
    st.header("Classify Email")
    model, vectorizer = load_model('spam_classifier')
    if model:
        text_input = st.text_area("Enter email text:", height=200)
        if st.button("Classify") and text_input:
            processed_text = preprocess_text(text_input)
            text_vectorized = vectorizer.transform([processed_text])
            prediction, probability = model.predict(text_vectorized)[0], model.predict_proba(text_vectorized)[0]
            st.markdown(f"### Classification: {'SPAM' if prediction else 'HAM'}")
            st.progress(probability.max())
            st.write(f"HAM: {probability[0]:.2%}, SPAM: {probability[1]:.2%}")
    else:
        st.info("No trained model found. Please train a model first.")

st.sidebar.header("About")
st.sidebar.write("This app allows you to train a spam classification model or classify emails using a trained model.")
st.sidebar.header("Instructions")
st.sidebar.write("Use the 'Train Model' tab to train a new model or the 'Classify Email' tab to classify text.")
