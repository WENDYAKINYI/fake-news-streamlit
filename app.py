import streamlit as st
import requests
import joblib
from newspaper import Article, Config
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ====================== CONFIGURATION ======================
TRUSTED_DOMAINS = [
    'bbc.com', 'cnn.com', 'reuters.com', 'apnews.com', 'nytimes.com',
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

# ====================== STREAMLIT UI ======================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Your favorite blue header
st.markdown("""
    <div style="background-color:#002B5B;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:white;text-align:center;font-family:Helvetica;">
            🧠 Fake News Detector
        </h1>
        <p style="color:white;text-align:center;">Paste an article or URL to check if it's real or fake.</p>
    </div>
""", unsafe_allow_html=True)

# Improved sidebar from newer version
with st.sidebar:
    st.markdown("**About This Tool**")
    st.markdown("""
    - AI/ML Model: Logistic Regression with TF-IDF
    - Accuracy: 94.7% on test data
    - Training Data: 44,889 political articles
    """)
    st.divider()
    st.caption("Model version: 2.1 | Last updated: June 2024")

# Tabbed interface from newer version
tab1, tab2 = st.tabs(["📝 Paste Article Text", "🔗 Enter Article URL"])
text_input = ""

with tab1:
    text_input = st.text_area(
        "Paste your article content here:",
        height=250,
        placeholder="Copy and paste the full text of the news article...",
        help="For best results, paste complete articles with multiple paragraphs"
    )

with tab2:
    url_input = st.text_input(
        "Enter news article URL:",
        placeholder="https://example.com/news-article",
        help="We'll extract text automatically from most news websites"
    )
    if url_input:
        with st.spinner("🔄 Processing URL..."):
            # Domain reputation check first
            domain = url_input.split('/')[2].replace('www.', '').lower()
            if domain in TRUSTED_DOMAINS:
                st.success("✅ Trusted news source detected")
                st.stop()
            elif domain in SUSPICIOUS_DOMAINS:
                st.error("🚨 Known unreliable source detected")
                st.stop()
                
            article_text, method = extract_article_content(url_input)
            if article_text:
                text_input = article_text
                st.success(f"✅ Successfully extracted article using {method}!")
                st.text_area("📄 Extracted Article Text:", text_input, height=200)
            else:
                st.error("❌ Failed to extract article content")

# Enhanced prediction section
if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some article text first.")
    else:
        with st.spinner("🧠 Analyzing content..."):
            prediction, confidence = classify_content(text_input)
            
            st.markdown("---")
            st.subheader("🧾 Prediction Results")

            if confidence < 0.65:
                st.warning(f"⚠️ Uncertain (Confidence: {confidence:.0%})")
            else:
                if prediction == "REAL":
                    st.success(f"✅ REAL NEWS — Confidence: {confidence:.0%}")
                else:
                    st.error(f"🚨 FAKE NEWS DETECTED — Confidence: {confidence:.0%}")
            
            # Improved confidence visualization
            st.markdown(f"""
            <div style="
                background: #f0f2f6;
                padding: 0.5rem;
                border-radius: 8px;
                margin: 1rem 0;
            ">
                <div style="
                    height: 8px; 
                    background: linear-gradient(90deg, #1E90FF {confidence*100}%, #eee {confidence*100}%);
                    margin-top: 0.5rem;
                "></div>
            </div>
            """, unsafe_allow_html=True)

            # Enhanced explanation
            with st.expander("ℹ️ Detailed Analysis", expanded=True):
                if prediction == "REAL":
                    st.markdown("""
                    **Characteristics of authentic content:**
                    - Balanced language with minimal sensationalism
                    - References to verifiable sources
                    - Moderate emotional tone
                    """)
                else:
                    st.markdown("""
                    **Warning signs detected:**
                    - Emotional/exaggerated language
                    - Lack of credible references
                    - Patterns common in misinformation
                    """)
                
                st.markdown("""
                **Recommended actions:**
                1. Cross-check with fact-checking resources
                2. Compare with other reputable sources
                3. Verify publication date and author
                """)

            # Fact-checking resources from newer version
            st.markdown("### 🔍 Verify With Trusted Sources")
            cols = st.columns(3)
            with cols[0]:
                st.link_button("FactCheck.org", "https://www.factcheck.org/")
            with cols[1]:
                st.link_button("Snopes", "https://www.snopes.com/")
            with cols[2]:
                st.link_button("Google Fact Check", "https://toolbox.google.com/factcheck/explorer")

# Footer note
st.markdown("---")
st.caption("Note: This tool provides algorithmic estimates. Always verify important claims through multiple sources.")
