import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import time
from huggingface_hub import login

# Constants
MODEL_NAME = "PSantaniello95/FAKE_NEWS_DETECTOR"  # Your Hugging Face model name
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the model and tokenizer from Hugging Face Hub"""
    try:
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Move model to device
        model.to(DEVICE)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_text(text, tokenizer):
    """Preprocess the input text"""
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs.to(DEVICE)

def predict(model, tokenizer, text):
    """Make prediction on the input text"""
    with torch.no_grad():
        inputs = preprocess_text(text, tokenizer)
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
    return {
        "prediction": "Fake" if prediction == 1 else "Real",
        "confidence": confidence,
        "probabilities": probabilities[0].cpu().numpy()
    }

def plot_prediction(probabilities):
    """Plot the prediction probabilities"""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Real", "Fake"]
    colors = ["#2ecc71", "#e74c3c"]
    
    bars = ax.bar(labels, probabilities, color=colors)
    
    # Add probability values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    
    return fig

def main():
    st.title("📰 Fake News Detector")
    st.markdown("""
    This app uses a fine-tuned RoBERTa model to detect fake news. 
    Enter a news article or text below to analyze its authenticity.
    """)
    
    # Load model
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please try again later.")
        return
    
    # Input text
    text = st.text_area("Enter the news article or text to analyze:", height=200)
    
    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Analyzing..."):
            # Make prediction
            result = predict(model, tokenizer, text)
            
            # Display results
            st.subheader("Analysis Results")
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction card
                st.markdown("### Prediction")
                prediction_color = "red" if result["prediction"] == "Fake" else "green"
                st.markdown(f"""
                <div style='background-color: {prediction_color}; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                    <h2 style='margin: 0;'>{result["prediction"]}</h2>
                    <p style='margin: 0;'>Confidence: {result["confidence"]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Probability plot
                st.markdown("### Probability Distribution")
                fig = plot_prediction(result["probabilities"])
                st.pyplot(fig)
            
            # Additional information
            st.markdown("### How to Interpret Results")
            st.markdown("""
            - **Real**: The model believes the text is likely to be authentic news
            - **Fake**: The model believes the text might be fake or misleading
            
            The confidence score indicates how certain the model is about its prediction.
            """)
            
            # Disclaimer
            st.markdown("""
            ---
            **Disclaimer**: This tool is for informational purposes only. 
            The results should not be considered as absolute truth and should be used 
            in conjunction with other fact-checking methods.
            """)

if __name__ == "__main__":
    main() 