"""
Streamlit Web Application for Bias Detection
Deploys the trained models as a web interface.

Author: AI Assistant
Date: 2024

To run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from data_preprocessing import TextPreprocessor
from model_lstm import LSTMBasedClassifier

# Page Configuration
st.set_page_config(
    page_title="Bias Detection AI",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5
    }
    .stTextArea textarea {
        font-size: 18px
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .biased {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .non-biased {
        background-color: #ccffcc;
        border-left: 5px solid #00ff00;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load ML and LSTM models"""
    
    # Load ML model
    ml_vectorizer = pickle.load(open('ml_vectorizer.pkl', 'rb'))
    lr_model = pickle.load(open('ml_logistic_regression.pkl', 'rb'))
    
    # Load LSTM model
    try:
        lstm_model = LSTMBasedClassifier()
        lstm_model.load_model('lstm_bias_model.h5')
    except:
        lstm_model = None
        st.warning("LSTM model not found. Using ML model only.")
    
    return (ml_vectorizer, lr_model), lstm_model


def predict_ml(text, ml_vectorizer, lr_model, preprocessor):
    """Predict using ML model"""
    cleaned = preprocessor.preprocess(text)
    text_tfidf = ml_vectorizer.transform([cleaned])
    prob = lr_model.predict_proba(text_tfidf)[0][1]
    
    prediction = 'Biased' if prob > 0.5 else 'Non-Biased'
    confidence = prob if prob > 0.5 else (1 - prob)
    
    # Get important words
    feature_names = ml_vectorizer.get_feature_names_out()
    coefficients = lr_model.coef_[0]
    
    words = cleaned.split()
    biased_words = []
    for word in words:
        if word in feature_names:
            idx = list(feature_names).index(word)
            if coefficients[idx] > 0:
                biased_words.append((word, coefficients[idx]))
    
    biased_words.sort(key=lambda x: x[1], reverse=True)
    
    return prediction, confidence * 100, biased_words[:5], prob


def predict_lstm(text, lstm_model):
    """Predict using LSTM model"""
    result = lstm_model.predict(text)
    return result['prediction'], result['confidence'], result['probability_bias']


# ======================
# MAIN APP
# ======================

def main():
    # Header
    st.title("🔍 AI Bias Detector")
    st.markdown("### Detect Bias in Text using Machine Learning & Deep Learning")
    st.markdown("---")
    
    # Load models
    ml_models, lstm_model = load_models()
    ml_vectorizer, lr_model = ml_models
    preprocessor = TextPreprocessor()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Logistic Regression (ML)", "LSTM (Deep Learning)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About:**
    This AI analyzes text to detect potential bias. 
    The model was trained on news articles and social media posts.
    
    **Note:** This is for educational purposes only.
    """)
    
    # Main Input
    st.subheader("📝 Enter Text to Analyze")
    text_input = st.text_area(
        "Paste your text here:",
        height=150,
        placeholder="Type or paste text here to check for bias..."
    )
    
    # Analyze Button
    if st.button("🚀 Analyze Text", type="primary"):
        if text_input:
            with st.spinner("Analyzing..."):
                
                try:
                    # Choose model
                    if "Logistic" in model_choice:
                        pred, conf, biased_words, prob = predict_ml(
                            text_input, ml_vectorizer, lr_model, preprocessor
                        )
                    else:
                        if lstm_model is None:
                            st.error("LSTM model not available!")
                            return
                        pred, conf, prob = predict_lstm(text_input, lstm_model)
                        biased_words = []
                    
                    # Display Results
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    # Color coding
                    box_class = "biased" if pred == "Biased" else "non-biased"
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-box {box_class}">
                            <h2>Prediction: {pred}</h2>
                            <p>Confidence: {conf:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar for confidence
                        st.progress(conf/100)
                    
                    with col2:
                        # Probability meter
                        st.metric("Probability of Bias", f"{prob*100:.2f}%")
                        st.metric("Probability of Non-Biased", f"{(1-prob)*100:.2f}%")
                    
                    # Show biased words if any
                    if biased_words:
                        st.markdown("### ⚠️ Potentially Biased Words")
                        for word, score in biased_words:
                            st.write(f"- **{word}** (score: {score:.4f})")
                    
                    # Tips
                    if pred == "Biased":
                        st.warning("💡 The text contains potentially biased language. Consider reviewing the word choice.")
                    else:
                        st