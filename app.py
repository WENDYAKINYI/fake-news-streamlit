import streamlit as st
import requests
import joblib
from newspaper import Article, Config
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ====================== CONFIGURATION ======================
TRUSTED_DOMAINS = [
    'bbc.com', 'reuters.com', 'apnews.com', 'nytimes.com',
    'washingtonpost.com', 'theguardian.com', 'nature.com',
    'science.org', 'who.int', 'nih.gov'
]

SUSPICIOUS_DOMAINS = [
    'infowars.com', 'naturalnews.com', 'beforeitsnews.com',
    'worldtruth.tv', 'yournewswire.com', 'newsbusters.org'
]

# ====================== CONTENT EXTRACTION ======================
def extract_article_content(url):
    """Enhanced content extraction with multiple fallbacks"""
    config = Config()
    config.browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    config.request_timeout = 10
    
    # Attempt 1: newspaper3k (best for news sites)
    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        if len(article.text.split()) > 50:  # Minimum 50 words
            return article.text, "newspaper3k"
    except Exception as e:
        st.warning(f"Primary extraction failed: {str(e)}")
    
    # Attempt 2: BeautifulSoup (fallback)
    try:
        headers = {"User-Agent": config.browser_user_agent}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try common article containers first
        for selector in ['article', 'main', '.article-body', '.post-content']:
            elements = soup.select(selector)
            if elements:
                text = ' '.join([e.get_text(separator=' ', strip=True) for e in elements])
                if len(text.split()) > 50:
                    return text, "beautifulsoup (article tag)"
        
        # Fallback to paragraph collection
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
    """Load or create model with proper error handling"""
    try:
        model = joblib.load("logistic_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        # Create minimal fallback model
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(
            TfidfVectorizer(max_features=5000),
            LogisticRegression(max_iter=1000)
        )
        return model, None

def classify_content(text):
    """Classify text with confidence scoring"""
    model, vectorizer = load_model()
    
    if vectorizer:  # Your custom model
        features = vectorizer.transform([text])
    else:  # Fallback model
        features = model.named_steps['tfidfvectorizer'].transform([text])
    
    proba = model.predict_proba(features)[0]
    confidence = max(proba)
    label = "REAL" if proba[0] > proba[1] else "FAKE"
    
    if confidence < 0.6:
        return "UNCERTAIN", confidence
    return label, confidence

# ====================== ORIGINAL STREAMLIT UI ======================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Original header styling
st.markdown("""
    <div style="background-color:#002B5B;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:white;text-align:center;font-family:Helvetica;">
            🧠 Fake News Detector
        </h1>
        <p style="color:white;text-align:center;">Paste an article or URL to check if it's real or fake.</p>
    </div>
""", unsafe_allow_html=True)

# Original sidebar
with st.sidebar:
    st.markdown("**Theme:** Fake vs Real News")
    st.markdown("**Stack:** Streamlit, newspaper3k, BeautifulSoup, scikit-learn")

# Original input method
st.subheader("Select Input Method:")
input_choice = st.radio("Choose how to enter news:", ["Paste Text", "Paste URL"], horizontal=True)
text_input = ""

# Text Option
if input_choice == "Paste Text":
    text_input = st.text_area("📄 Paste article text here:", height=200)

# URL Option with fallback to BeautifulSoup
elif input_choice == "Paste URL":
    url_input = st.text_input("🔗 Paste article URL here:")
    if url_input:
        with st.spinner("Fetching article..."):
            # First check domain reputation
            domain = url_input.split('/')[2].replace('www.', '').lower()
            if domain in TRUSTED_DOMAINS:
                st.success("✅ Trusted news source detected")
                st.stop()
            elif domain in SUSPICIOUS_DOMAINS:
                st.error("🚨 Known unreliable source detected")
                st.stop()
                
            article_text, method = extract_article_content(url_input)
            if article_text:
                st.success(f"✅ Extracted with {method}!")
                text_input = st.text_area("📄 Extracted Article Text:", article_text, height=200)
            else:
                st.error("❌ Failed to extract article content")

# Prediction with original styling
if st.button("🔍 Analyze"):
    if not text_input.strip():
        st.warning("Please enter some article text first.")
    else:
        with st.spinner("🧠 Analyzing..."):
            prediction, confidence = classify_content(text_input)
            
            st.markdown("---")
            st.subheader("🧾 Prediction Results")

            if confidence < 0.65:
                st.warning(f"⚠️ Model is uncertain (Confidence: {confidence:.0%})")
            else:
                if prediction == "REAL":
                    st.success(f"✅ REAL NEWS — Confidence: {confidence:.0%}")
                else:
                    st.error(f"🚨 FAKE NEWS DETECTED — Confidence: {confidence:.0%}")
            
            st.progress(int(confidence * 100))

            with st.expander("ℹ️ What does this mean?"):
                if prediction == "REAL":
                    st.markdown("""
                    - Writing style matches verified news sources
                    - Contains balanced perspectives
                    - Shows characteristics of factual reporting
                    """)
                else:
                    st.markdown("""
                    - Shows signs of potential misinformation:
                      - Sensational or exaggerated language
                      - Lacks credible sources
                      - Contains bias patterns
                    """)

            st.markdown("### Next Steps")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.button("📰 Analyze Another")
            with c2:
                st.link_button("🔍 FactCheck.org", "https://www.factcheck.org/")
            with c3:
                st.link_button("📚 Learn More", "https://medialiteracynow.org/")
