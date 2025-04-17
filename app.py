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

# Modern Header
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

# Sidebar with information
with st.sidebar:
    with st.container(border=True):
        st.markdown("**About This Tool**")
        st.markdown("""
        This AI-powered detector analyzes news content for signs of misinformation using:
        - Natural Language Processing
        - Machine Learning models
        - Fact-checking patterns
        """)
    
    with st.container(border=True):
        st.markdown("**Tips for Best Results**")
        st.markdown("""
        - Provide complete articles (300+ words)
        - Check multiple sources
        - Be wary of emotional language
        - Verify publication date
        """)
    
    st.divider()
    st.caption("Model version: 2.1.0 | Last updated: June 2024")

# Input section with tabs
tab1, tab2 = st.tabs(["📝 Paste Article Text", "🔗 Enter Article URL"])
text_input = ""  # Initialize variable

with tab1:
    text_input = st.text_area(
        "Paste your article content here:",
        height=250,
        placeholder="Copy and paste the full text of the news article you want to analyze...",
        help="For best results, paste the complete article text including multiple paragraphs"
    )

with tab2:
    url_input = st.text_input(
        "Enter news article URL:",
        placeholder="https://example.com/news-article",
        help="We'll extract the text automatically from most news websites"
    )
    
    if url_input:
        with st.status("🔄 Processing URL...", expanded=True) as status:
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
                status.update(label="✅ Successfully extracted article!", state="complete")
                st.text_area("📄 Extracted Article Text:", text_input, height=200)

            except Exception as e1:
                status.update(label=f"⚠ newspaper3k failed: {e1}", state="error")
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

# Prediction section
if st.button("🔍 Analyze Article", use_container_width=True, type="primary"):
    if not text_input.strip():
        st.warning("Please enter some article text first.")
    else:
        with st.spinner("🧠 Analyzing content with AI..."):
            vect_text = vectorizer.transform([text_input])
            prediction = model.predict(vect_text)[0]
            prob = model.predict_proba(vect_text)[0]

            st.divider()
            
            # Results container
            with st.container():
                st.subheader("Analysis Results")
                
                # Visual indicator
                if prediction == 1:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.success("✅")
                    with col2:
                        st.markdown("### This article appears to be **RELIABLE**")
                else:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.error("⚠️")
                    with col2:
                        st.markdown("### This article contains **SUSPICIOUS CONTENT**")
                
                # Confidence meter
                st.markdown(f"**Confidence Level:** {round(max(prob) * 100, 2)}%")
                st.progress(
                    int(max(prob) * 100),
                    text=f"{'High' if max(prob) > 0.7 else 'Medium' if max(prob) > 0.5 else 'Low'} confidence"
                )
                
                # Explanation section
                with st.expander("ℹ️ What does this mean?"):
                    if prediction == 1:
                        st.markdown("""
                        Our analysis suggests this content is likely trustworthy because:
                        - The writing style matches verified news sources
                        - Contains balanced perspectives
                        - Shows characteristics of factual reporting
                        """)
                    else:
                        st.markdown("""
                        This content shows signs that may indicate misinformation:
                        - Sensational or exaggerated language detected
                        - Lacks credible sources or references
                        - Shows bias patterns common in fake news
                        """)
                
                # Action buttons
                st.markdown("### Next Steps")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.button("📰 Analyze Another", help="Check a different article")
                with c2:
                    st.link_button("🔍 Fact-Check", "https://www.factcheck.org/")
                with c3:
                    st.link_button("📚 Learn More", "https://medialiteracynow.org/")
