import streamlit as st
import joblib
import re
import nltk
import gdown
import os

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download model
if not os.path.exists("model.pkl"):
    url = "https://drive.google.com/uc?id=1MxSgZAhPPqj9Up3q57WFuDWLqG0jNKr2"
    gdown.download(url, "model.pkl", quiet=False)

# Load files
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title("Spam Detection App")

user_input = st.text_area("Enter message")

if st.button("Predict"):
    if user_input:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")
    else:
        st.warning("Enter message")
