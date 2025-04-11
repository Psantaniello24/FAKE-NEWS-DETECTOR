import streamlit as st
import requests
import json
import pandas as pd
from typing import List, Dict
import plotly.express as px

# Configure the page
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Title and description
st.title("📰 Fake News Detector")
st.markdown("""
This application uses a fine-tuned BERT/RoBERTa model to detect fake news.
Enter a news article text below to analyze it.
""")

# API endpoint
API_URL = "http://localhost:8000"

def predict_single(text: str) -> Dict:
    """Make a prediction for a single text using the API."""
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text}
    )
    return response.json()

def predict_batch(texts: List[str]) -> List[Dict]:
    """Make predictions for multiple texts using the API."""
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"texts": texts}
    )
    return response.json()["predictions"]

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Single Article", "Batch Analysis"])

with tab1:
    # Single article analysis
    st.header("Analyze Single Article")
    
    # Text input
    article_text = st.text_area(
        "Enter the article text:",
        height=200,
        placeholder="Paste your article text here..."
    )
    
    if st.button("Analyze"):
        if article_text:
            with st.spinner("Analyzing article..."):
                try:
                    result = predict_single(article_text)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Prediction",
                            "Real News" if result["prediction"] == 0 else "Fake News",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Confidence",
                            f"{result['confidence']:.2%}",
                            delta=None
                        )
                    
                    # Add a color-coded result
                    color = "green" if result["prediction"] == 0 else "red"
                    st.markdown(
                        f"<h3 style='color: {color};'>"
                        f"{'✅ This appears to be REAL NEWS' if result['prediction'] == 0 else '⚠️ This appears to be FAKE NEWS'}"
                        "</h3>",
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    st.error(f"Error analyzing article: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    # Batch analysis
    st.header("Batch Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV file with articles (must have a 'text' column)",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if "text" not in df.columns:
                st.error("CSV file must contain a 'text' column")
            else:
                if st.button("Analyze Batch"):
                    with st.spinner("Analyzing articles..."):
                        try:
                            results = predict_batch(df["text"].tolist())
                            
                            # Create results DataFrame
                            results_df = pd.DataFrame(results)
                            results_df["prediction"] = results_df["prediction"].map({0: "Real", 1: "Fake"})
                            
                            # Display summary statistics
                            st.subheader("Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Total Articles",
                                    len(results_df)
                                )
                            
                            with col2:
                                fake_count = len(results_df[results_df["prediction"] == "Fake"])
                                st.metric(
                                    "Fake News Count",
                                    fake_count
                                )
                            
                            with col3:
                                real_count = len(results_df[results_df["prediction"] == "Real"])
                                st.metric(
                                    "Real News Count",
                                    real_count
                                )
                            
                            # Display distribution plot
                            fig = px.pie(
                                results_df,
                                names="prediction",
                                title="Distribution of Predictions"
                            )
                            st.plotly_chart(fig)
                            
                            # Display detailed results
                            st.subheader("Detailed Results")
                            st.dataframe(
                                results_df[["text", "prediction", "confidence"]]
                                .sort_values("confidence", ascending=False)
                            )
                            
                        except Exception as e:
                            st.error(f"Error analyzing batch: {str(e)}")
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using BERT/RoBERTa and FastAPI</p>
    <p>Note: This is a demo application. The model's predictions should not be considered as absolute truth.</p>
</div>
""", unsafe_allow_html=True) 