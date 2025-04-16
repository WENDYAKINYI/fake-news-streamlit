import streamlit as st
import joblib
from newspaper import Article

# Load model and vectorizer
model = joblib.load("model.pkl")         # Your trained model
vectorizer = joblib.load("vectorizer.pkl")  # Your TF-IDF vectorizer

# App title
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Paste a news article **text** or **URL** to check if it's real or fake.")

# Choose input type
st.subheader("Input Method:")
input_choice = st.radio("Select input type:", ["Text", "URL"])

text_input = ""

# Text input option
if input_choice == "Text":
    text_input = st.text_area("Paste article text:", height=200)

# URL input option
elif input_choice == "URL":
    url_input = st.text_input("Paste article URL here:")
    if url_input:
        try:
            article = Article(url_input)
            article.download()
            article.parse()
            if not article.text.strip():
                raise ValueError("No article text found.")
            text_input = article.text
            st.success("✅ Article extracted successfully!")
            st.text_area("Extracted article text:", text_input, height=200)
        except Exception as e:
            st.error(f"❌ Failed to extract article: {str(e)}")

# Prediction
if st.button("Check if it's Fake or Real"):
    if not text_input.strip():
        st.warning("Please enter or extract some text first.")
    else:
        # Transform and predict
        vect_text = vectorizer.transform([text_input])
        prediction = model.predict(vect_text)[0]
        prob = model.predict_proba(vect_text)[0]
        label = "🟢 REAL NEWS" if prediction == 1 else "🔴 FAKE NEWS"
        st.markdown(f"## Prediction: {label}")
        st.markdown(f"**Confidence:** {round(max(prob) * 100, 2)}%")


