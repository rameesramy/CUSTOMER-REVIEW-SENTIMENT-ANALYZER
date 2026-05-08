
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# --- UI ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("💬 Customer Review Sentiment Analyzer")
st.markdown("Enter a customer review below and the AI will predict its sentiment.")

user_input = st.text_area("📝 Your Review", placeholder="Type your review here...", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == "Positive":
            st.success("✅ Sentiment: **Positive** 😊")
        elif prediction == "Negative":
            st.error("❌ Sentiment: **Negative** 😞")
        else:
            st.info("➖ Sentiment: **Neutral** 😐")

st.markdown("---")
st.caption("Built with scikit-learn + Streamlit | TCS Industry Project")
