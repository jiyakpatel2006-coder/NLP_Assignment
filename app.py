import streamlit as st
import joblib
import re
import nltk

# Download NLTK data safely
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# LOAD FILES SAFELY
# -------------------------------
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error("❌ Model files not found. Make sure model.pkl and vectorizer.pkl are uploaded.")
    st.stop()

# -------------------------------
# NLP SETUP
# -------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# -------------------------------
# CLEAN FUNCTION (SAFE)
# -------------------------------
def clean_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        return " ".join(words)
    except:
        return ""

# -------------------------------
# UI
# -------------------------------
st.title("📩 Spam Detection App")
st.write("Enter a message to check if it's spam")

user_input = st.text_area("Enter message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        if str(prediction[0]) in ["1", "spam"]:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
