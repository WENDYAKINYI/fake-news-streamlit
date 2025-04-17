import streamlit as st
import joblib
from newspaper import Article, Config
from bs4 import BeautifulSoup
import requests

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Header
# Replace your current header with this more modern version
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        .main-header {
            background: linear-gradient(135deg, #002B5B 0%, #1A5F7A 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            color: white;
            text-align: center;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .main-header p {
            color: rgba(255,255,255,0.9);
            text-align: center;
            font-family: 'Poppins', sans-serif;
            font-size: 1rem;
        }
    </style>
    <div class="main-header">
        <h1>🧠 AI-Powered Fake News Detector</h1>
        <p>Analyze news articles for authenticity with our advanced detection system</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("**Theme:** Fake vs Real News")
    st.markdown("**Stack:** Streamlit, newspaper3k, BeautifulSoup, scikit-learn")

# Input method
st.subheader("Select Input Method:")
input_choice = st.radio("Choose how to enter news:", ["Paste Text", "Paste URL"])
text_input = ""

# Text Option
if input_choice == "Paste Text":
    text_input = st.text_area("📄 Paste article text here:", height=200)

# URL Option with fallback to BeautifulSoup
elif input_choice == "Paste URL":
    url_input = st.text_input("🔗 Paste article URL here:")
    if url_input:
        with st.spinner("Fetching article..."):
            try:
                config = Config()
                config.browser_user_agent = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                )
                article = Article(url_input, config=config)
                article.download()
                article.parse()
                if not article.text.strip():
                    raise ValueError("No article text found.")
                text_input = article.text
                st.success("✅ Extracted with newspaper3k!")
                st.text_area("📄 Extracted Article Text:", text_input, height=200)

            except Exception as e1:
                st.warning(f"⚠ newspaper3k failed: {e1}")
                st.info("🔄 Trying fallback method...")
                try:
                    headers = {"User-Agent": config.browser_user_agent}
                    res = requests.get(url_input, headers=headers)
                    soup = BeautifulSoup(res.content, "html.parser")
                    paragraphs = soup.find_all("p")
                    article_text = " ".join(p.get_text() for p in paragraphs)
                    if not article_text.strip():
                        raise ValueError("No text found.")
                    text_input = article_text
                    st.success("✅ Extracted with BeautifulSoup!")
                    st.text_area("📄 Extracted Article Text:", text_input, height=200)
                except Exception as e2:
                    st.error(f"❌ Failed to extract using fallback. {e2}")

# Prediction
if st.button("🔍 Analyze"):
    if not text_input.strip():
        st.warning("Please enter some article text first.")
    else:
        with st.spinner("🧠 Analyzing..."):
            vect_text = vectorizer.transform([text_input])
            prediction = model.predict(vect_text)[0]
            prob = model.predict_proba(vect_text)[0]

            st.markdown("---")
            st.subheader("🧾 Prediction Results")

            if prediction == 1:
                st.success("✅ **REAL NEWS** — This appears trustworthy.")
            else:
                st.error("🚨 **FAKE NEWS DETECTED** — Be cautious sharing this.")

            st.markdown(f"**Confidence:** {round(max(prob) * 100, 2)}%")
            st.progress(int(max(prob) * 100))

