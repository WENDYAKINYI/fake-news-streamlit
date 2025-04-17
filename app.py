import streamlit as st
import joblib
from newspaper import Article, Config
from bs4 import BeautifulSoup
import requests

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
            # Try with newspaper3k first
            user_config = Config()
            user_config.browser_user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
            article = Article(url_input, config=user_config)
            article.download()
            article.parse()
            article_text = article.text

            if not article_text.strip():
                raise ValueError("No text found from newspaper3k.")

            text_input = article_text
            st.success("✅ Extracted with newspaper3k!")
            st.text_area("Extracted article text:", text_input, height=200)

        except Exception as e1:
            st.warning(f"⚠ newspaper3k failed: {e1}")
            st.info("🔄 Trying fallback extraction...")

            try:
                headers = {
                    "User-Agent": user_config.browser_user_agent
                }
                response = requests.get(url_input, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")

                # Get all paragraphs
                paragraphs = soup.find_all("p")
                article_text = " ".join(p.get_text() for p in paragraphs)

                if not article_text.strip():
                    raise ValueError("Fallback extraction found no usable text.")

                text_input = article_text
                st.success("✅ Extracted with BeautifulSoup fallback!")
                st.text_area("Extracted article text:", text_input, height=200)

            except Exception as e2:
                st.error(f"❌ Failed to extract article from URL. Error: {e2}")


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


