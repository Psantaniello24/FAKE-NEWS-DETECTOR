import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Tuple
import pandas as pd

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text: str) -> str:
    """
    Clean the input text by removing URLs, special characters, and extra whitespace.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from the text.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_text(text: str) -> str:
    """
    Apply all preprocessing steps to the text.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    return text

def load_and_preprocess_data(file_path: str) -> Tuple[List[str], List[int]]:
    """
    Load data from CSV and preprocess the text column.
    """
    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must contain 'text' and 'label' columns")
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    return df['processed_text'].tolist(), df['label'].tolist() 