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
        st.success("Loaded your trained model")
        return model, vectorizer
    except Exception as e:
        st.warning(f"Model loading failed: {str(e)}. Using fallback.")
        
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
    page_title="Universal News Verifier",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTextArea textarea { min-height: 200px; }
    .stProgress > div > div > div { background-color: #1E90FF; }
    .st-eb { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This tool verifies news authenticity using:
    - **Domain reputation** checks
    - **Content analysis** (machine learning)
    - **Multi-source extraction**
    """)
    st.divider()
    st.markdown("**Trusted Sources:**")
    st.caption(", ".join(TRUSTED_DOMAINS[:5]) + "...")
    st.markdown("**Suspicious Sources:**")
    st.caption(", ".join(SUSPICIOUS_DOMAINS[:3]) + "...")
    st.divider()
    st.caption("Version 2.1 | Last updated: June 2024")

# Main UI
st.title("🔍 Universal News Verifier")
st.caption("Analyze any news article or text snippet")

# Input options
tab1, tab2 = st.tabs(["📝 Paste Text", "🔗 Enter URL"])

with tab1:
    text_content = st.text_area("Paste article text:", help="Minimum 100 characters for best results")

with tab2:
    url_input = st.text_input("Enter article URL:", placeholder="https://example.com/news-article")
    if url_input:
        with st.spinner("Extracting article content..."):
            article_text, method = extract_article_content(url_input)
            if article_text:
                st.success(f"✅ Extracted using {method}")
                text_content = st.text_area("Extracted Content", article_text, height=250)
            else:
                st.error("Failed to extract meaningful content")

# Analysis button
if st.button("Analyze Authenticity", type="primary", use_container_width=True):
    if not text_content or len(text_content.split()) < 20:
        st.warning("Please provide sufficient text (at least 20 words)")
        st.stop()
    
    # Domain verification for URLs
    if 'url_input' in locals() and url_input:
        domain = url_input.split('/')[2].replace('www.', '').lower()
        if domain in TRUSTED_DOMAINS:
            st.success("""
            ## ✅ Verified Trusted Source
            *This domain is on our pre-approved list of reliable sources*
            """)
            st.stop()
        elif domain in SUSPICIOUS_DOMAINS:
            st.error("""
            ## ❌ Known Unreliable Source
            *This domain is on our list of frequently misleading sites*
            """)
            st.stop()
    
    # Content analysis
    with st.spinner("Analyzing content..."):
        label, confidence = classify_content(text_content)
    
    # Display results
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if label == "UNCERTAIN":
            st.warning(f"## ⚠️ Inconclusive Result")
        elif label == "REAL":
            st.success(f"## ✅ Likely Authentic")
        else:
            st.error(f"## ❌ Likely Misleading")
    
    with col2:
        st.metric("Confidence", f"{confidence:.0%}")
        st.progress(confidence)
    
    # Explanation
    with st.expander("Detailed Analysis", expanded=True):
        if label == "UNCERTAIN":
            st.markdown("""
            The model couldn't determine with sufficient confidence because:
            - The writing style is ambiguous
            - Contains mixed characteristics
            - May be on an unfamiliar topic
            """)
        elif label == "REAL":
            st.markdown("""
            Characteristics of authentic content:
            - Balanced language
            - Credible sources cited
            - Moderate emotional tone
            """)
        else:
            st.markdown("""
            Warning signs detected:
            - Sensational/exaggerated language
            - Lack of verifiable sources
            - Emotional manipulation cues
            """)
        
        st.markdown("""
        ### Recommended Actions:
        1. Check with additional sources below
        2. Verify author/publisher reputation
        3. Look for corroborating evidence
        """)
    
    # Verification resources
    st.markdown("---")
    st.subheader("🔍 Fact-Checking Resources")
    cols = st.columns(3)
    with cols[0]:
        st.link_button("FactCheck.org", "https://www.factcheck.org/")
    with cols[1]:
        st.link_button("Snopes", "https://www.snopes.com/")
    with cols[2]:
        st.link_button("Google Fact Check", "https://toolbox.google.com/factcheck/explorer")

# Footer
st.markdown("---")
st.caption("""
*Note: This tool provides algorithmic estimates, not definitive truth. Always use critical thinking.*
""")
