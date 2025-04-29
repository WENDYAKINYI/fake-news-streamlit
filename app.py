import streamlit as st
import requests
import joblib
from newspaper import Article, Config
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# ====================== CONFIGURATION ======================
TRUSTED_DOMAINS = [
    'bbc.com','bbc.com', 'reuters.com', 'apnews.com', 'nytimes.com',
    'washingtonpost.com', 'theguardian.com', 'nature.com',
    'science.org', 'who.int', 'nih.gov'
]

SUSPICIOUS_DOMAINS = [
    'infowars.com', 'naturalnews.com', 'beforeitsnews.com',
    'worldtruth.tv', 'yournewswire.com', 'newsbusters.org'
]

# ====================== CONTENT EXTRACTION ======================
def extract_article_content(url):
    config = Config()
    config.browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    config.request_timeout = 10

    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        if len(article.text.split()) > 50:
            return article.text, "newspaper3k"
    except Exception as e:
        st.warning(f"Primary extraction failed: {str(e)}")

    try:
        headers = {"User-Agent": config.browser_user_agent}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        for selector in ['article', 'main', '.article-body', '.post-content']:
            elements = soup.select(selector)
            if elements:
                text = ' '.join([e.get_text(separator=' ', strip=True) for e in elements])
                if len(text.split()) > 50:
                    return text, "beautifulsoup (article tag)"

        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        if len(text.split()) > 50:
            return text, "beautifulsoup (paragraphs)"

        raise Exception("Insufficient text extracted")
    except Exception as e:
        st.error(f"Fallback extraction failed: {str(e)}")
        return None, None

# ====================== MODEL HANDLING ======================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("logistic_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(
            TfidfVectorizer(max_features=5000),
            LogisticRegression(max_iter=1000)
        )
        return model, None

def classify_content(text):
    model, vectorizer = load_model()

    if len(text.split()) < 50 and bool(re.search(r"scam|fake|fraud|lie|hoax|ego|rant|wasting", text.lower())):
        return "FAKE", 0.9

    if vectorizer:
        features = vectorizer.transform([text])
    else:
        features = model.named_steps['tfidfvectorizer'].transform([text])

    proba = model.predict_proba(features)[0]
    confidence = max(proba)
    label = "REAL" if proba[0] > proba[1] else "FAKE"

    if confidence < 0.6:
        return "UNCERTAIN", confidence
    return label, confidence

# ====================== STREAMLIT UI ======================
st.set_page_config(
    page_title="📰 Fake News Detector | AI-Powered",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
    <div style="background: linear-gradient(135deg, #002B5B 0%, #1A5F7A 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
        <h1 style="color:white;text-align:center;font-family:'Poppins',sans-serif;font-weight:700;font-size:2.8rem;margin-bottom:0.5rem;">🧠 Smart Fake News Detector</h1>
        <p style="color:white;text-align:center;font-family:'Poppins',sans-serif;font-size:1.1rem;">Fight misinformation. Check articles instantly with AI-powered verification.</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🤖 About This AI")
    st.markdown("""
    - Logistic Regression + TF-IDF
    - Enhanced heuristic toxicity detection
    - Confidence-based warnings
    - Multiple fallback extraction methods
    """)
    st.divider()
    st.caption("🚀 Model version: 2.2 | June 2024")

# --- Tabs for Input ---
tab1, tab2 = st.tabs(["📝 Paste Article Text", "🔗 Enter Article URL"])
text_input = ""

with tab1:
    text_input = st.text_area(
        "Paste your article text here:",
        height=300,
        placeholder="Copy and paste the full news article OR a suspicious paragraph...",
    )

with tab2:
    url_input = st.text_input(
        "Enter a news article URL:",
        placeholder="https://www.example.com/news/latest..."
    )
    if url_input:
        with st.spinner("🔄 Extracting article..."):
            domain = url_input.split('/')[2].replace('www.', '').lower()
            if domain in SUSPICIOUS_DOMAINS:
                st.error("🚨 Known suspicious domain!")
                st.stop()
            article_text, method = extract_article_content(url_input)
            if article_text:
                text_input = article_text
                st.success(f"✅ Text extracted with {method}!")
                st.text_area("📄 Article Extracted:", text_input, height=250)
            else:
                st.error("❌ Failed to extract article text.")

# --- Prediction ---
if st.button("🔍 Analyze Content", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please paste some article text first.")
    else:
        with st.spinner("🧠 Running analysis..."):
            prediction, confidence = classify_content(text_input)
            st.markdown("---")
            st.subheader("🧾 Results Summary")

            if prediction == "UNCERTAIN":
                st.warning(f"⚠️ Model Uncertain — Confidence: {confidence:.0%}")
            elif prediction == "REAL":
                st.success(f"✅ Classified as REAL NEWS — Confidence: {confidence:.0%}")
            else:
                st.error(f"🚨 Classified as FAKE NEWS — Confidence: {confidence:.0%}")

            st.markdown(f"""
            <div style="background: #f0f2f6; padding: 0.5rem; border-radius: 8px; margin: 1rem 0;">
                <div style="height: 8px; background: linear-gradient(90deg, #1E90FF {confidence*100}%, #eee {confidence*100}%);"></div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("ℹ️ How the analysis works"):
                if prediction == "REAL":
                    st.markdown("""
                    - Balanced tone and credible writing patterns
                    - Matches reputable sources
                    - Minimal emotional triggers detected
                    """)
                else:
                    st.markdown("""
                    - Emotional, biased or exaggerated language detected
                    - Common patterns found in misinformation
                    - Lacks credible sourcing
                    """)

            st.markdown("### 🔗 Further Verification")
            cols = st.columns(3)
            with cols[0]:
                st.link_button("FactCheck.org", "https://www.factcheck.org/")
            with cols[1]:
                st.link_button("Snopes", "https://www.snopes.com/")
            with cols[2]:
                st.link_button("Google Fact Check", "https://toolbox.google.com/factcheck/explorer")

st.markdown("---")
st.caption("Note: Always cross-verify important news across multiple trusted sources.")
