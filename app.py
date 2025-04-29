import streamlit as st
import joblib
from newspaper import Article, Config
from bs4 import BeautifulSoup
import requests

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# --- Load Logistic Regression Model and TF-IDF Vectorizer ---
@st.cache_resource
def load_model():
    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

log_model, tfidf_vectorizer = load_model()

# --- Classify Text Using Logistic Regression ---
def classify_article(text):
    vect_text = tfidf_vectorizer.transform([text])
    prediction = log_model.predict(vect_text)[0]
    probabilities = log_model.predict_proba(vect_text)[0]
    return prediction, probabilities

# --- Extract Text from URL ---
def extract_text_from_url(url):
    try:
        config = Config()
        config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        article = Article(url, config=config)
        article.download()
        article.parse()
        if not article.text.strip():
            raise ValueError("No article text found.")
        return article.text
    except Exception:
        try:
            headers = {"User-Agent": config.browser_user_agent}
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.content, "html.parser")
            paragraphs = soup.find_all("p")
            return " ".join(p.get_text() for p in paragraphs)
        except Exception as e:
            return None

# --- UI Elements ---
st.markdown("""
    <div style="background: linear-gradient(135deg, #002B5B 0%, #1A5F7A 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
        <h1 style="color: white; text-align: center; font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 2.5rem;">🧠 AI-Powered Fake News Detector</h1>
        <p style="color: rgba(255,255,255,0.9); text-align: center; font-family: 'Poppins', sans-serif; font-size: 1rem;">Analyze news articles for authenticity in real time</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**About This Tool**")
    st.markdown("""
    - AI & ML based news authenticity detector
    - Capable of analyzing pasted text or extracted articles from URLs
    - Displays model certainty levels
    """)
    st.divider()
    st.caption("Model: Logistic Regression + TF-IDF | Updated: June 2024")

# --- Tabs for Input ---
tab1, tab2 = st.tabs(["📝 Paste Article Text", "🔗 Enter Article URL"])
text_input = ""

# Tab 1
with tab1:
    text_input = st.text_area("Paste article text:", height=250)

# Tab 2
with tab2:
    url_input = st.text_input("Enter article URL:")
    if url_input:
        with st.spinner("🔄 Extracting article text..."):
            article_text = extract_text_from_url(url_input)
            if article_text:
                text_input = article_text
                st.success("✅ Successfully extracted article!")
                st.text_area("Extracted article text:", text_input, height=200)
            else:
                st.error("❌ Failed to extract article text from the URL.")

# --- Prediction ---
if st.button("🔍 Analyze"):
    if not text_input.strip():
        st.warning("Please enter some article text or URL.")
    else:
        with st.spinner("🧠 Analyzing article authenticity..."):
            prediction, probabilities = classify_article(text_input)
            confidence = round(max(probabilities) * 100, 2)
            confidence_label = (
                "🟢 High Confidence" if confidence > 70 else
                "🟡 Medium Confidence" if confidence > 50 else
                "🔴 Low Confidence"
            )

            st.markdown("---")
            st.subheader("🧾 Prediction Results")

            if confidence < 65:
                st.warning(f"⚠️ Model is uncertain about this article (Confidence: {confidence}%). Please manually fact-check.")
            else:
                if prediction == 1:
                    st.success(f"✅ REAL NEWS — Confidence: {confidence}% ({confidence_label})")
                else:
                    st.error(f"🚨 FAKE NEWS DETECTED — Confidence: {confidence}% ({confidence_label})")

            st.progress(int(confidence))

            with st.expander("ℹ️ What Does This Mean?"):
                if prediction == 1:
                    st.markdown("""
                    - Writing style matches verified real news sources.
                    - Contains balanced and factual reporting patterns.
                    - Moderate emotional language.
                    """)
                else:
                    st.markdown("""
                    - Language patterns suggest bias, exaggeration, or misinformation.
                    - Lacks credible sourcing or shows signs of manipulation.
                    - Often uses sensational language.
                    """)

            st.markdown("### Trusted Fact-Checking Resources")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.link_button("🔍 FactCheck.org", "https://www.factcheck.org/")
            with c2:
                st.link_button("🕵️ Snopes", "https://www.snopes.com/")
            with c3:
                st.link_button("🌐 Google Chrome - Return to Web", "https://www.google.com/chrome/")
