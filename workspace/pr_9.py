import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    return " ".join(tokens)

# Cache dataset loading and preprocessing
@st.cache_data
def load_and_prepare_data():
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")
    true_df['label'] = 1
    fake_df['label'] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df[['text', 'label']]
    df['clean_text'] = df['text'].apply(clean_text)
    return df

# Cache model training
@st.cache_resource
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    return model, vectorizer, accuracy

# Load and preprocess data (cached)
df = load_and_prepare_data()

# Train model (cached)
model, vectorizer, accuracy = train_model(df)

# Streamlit UI setup
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown("""
<style>
body {
background-color: #0f0f0f;
color: #e0e0e0;
}
.main {
background-color: #0f0f0f;
}
h1 {
text-align: center;
color: #00ffae;
}
.stTextArea textarea {
background-color: #1e1e1e;
color: white;
}
.stButton > button {
color: white;
background: #00ffae;
border-radius: 8px;
padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.write(f"**Model Accuracy:** {accuracy:.2%}")

manual_text = st.text_area("Paste a news article text here 👇", height=300)

if st.button("Analyze Text"):
    if manual_text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing text..."):
            cleaned_input = clean_text(manual_text)
            transformed_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(transformed_input)[0]

            st.subheader("📝 Prediction Result:")
            if prediction == 1:
                st.success("✅ This news article appears to be *Real*.")
            else:
                st.error("🚨 This news article appears to be *Fake*.")
