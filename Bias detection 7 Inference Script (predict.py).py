"""
Prediction Script for Bias Detection
Loads trained models and makes predictions on new text.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Import custom modules
from data_preprocessing import TextPreprocessor


def predict_with_ml(text, model_prefix='ml'):
    """
    Predict using Traditional ML models.
    
    Args:
        text: Input text string
        model_prefix: Prefix used when saving models
        
    Returns:
        Prediction dictionary
    """
    # Load models
    ml_vectorizer = pickle.load(open(f'{model_prefix}_vectorizer.pkl', 'rb'))
    lr_model = pickle.load(open(f'{model_prefix}_logistic_regression.pkl', 'rb'))
    
    # Preprocess
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.preprocess(text)
    
    # Transform
    text_tfidf = ml_vectorizer.transform([cleaned_text])
    
    # Predict
    prob = lr_model.predict_proba(text_tfidf)[0][1]
    prediction = 'Biased' if prob > 0.5 else 'Non-Biased'
    confidence = prob if prob > 0.5 else (1 - prob)
    
    # Get feature importance for this prediction
    feature_names = ml_vectorizer.get_feature_names_out()
    coefficients = lr_model.coef_[0]
    
    # Find words in text that contributed most
    words_in_text = cleaned_text.split()
    biased_words_found = []
    
    for word in words_in_text:
        if word in feature_names:
            idx = list(feature_names).index(word)
            if coefficients[idx] > 0:
                biased_words_found.append((word, coefficients[idx]))
    
    # Sort by importance
    biased_words_found.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'model': 'Logistic Regression',
        'prediction': prediction,
        'confidence': round(confidence * 100, 2),
        'probability': {
            'Non-Biased': round((1-prob) * 100, 2),
            'Biased': round(prob * 100, 2)
        },
        'key_biased_words': biased_words_found[:5]
    }


def predict_with_lstm(text, model_path='lstm_bias_model.h5'):
    """
    Predict using LSTM model.
    
    Args:
        text: Input text string
        model_path: Path to saved LSTM model
        
    Returns:
        Prediction dictionary
    """
    from model_lstm import LSTMBasedClassifier
    
    # Load model
    lstm = LSTMBasedClassifier()
    lstm.load_model(model_path)
    
    # Predict
    result = lstm.predict(text)
    
    return {
        'model': 'LSTM',
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'probability': {
            'Non-Biased': round((1 - result['probability_bias']) * 100, 2),
            'Biased': round(result['probability_bias'] * 100, 2)
        }
    }


def predict_with_bert(text, model_path='bert_bias_model'):
    """
    Predict using BERT model.
    
    Args:
        text: Input text string
        model_path: Path to saved BERT model
        
    Returns:
        Prediction dictionary
    """
    from model_bert import BERTClassifier
    
    # Load model
    bert = BERTClassifier()
    bert.load_model(model_path)
    
    # Predict
    results = bert.predict(text)
    
    return {
        'model': 'BERT',
        'prediction': results[0]['prediction'],
        'confidence': results[0]['confidence'],
        'probability': results[0]['probabilities']
    }


def predict_bias(text, model_type='ml'):
    """
    Main prediction function.
    
    Args:
        text: Input text to analyze
        model_type: Type of model ('ml', 'lstm', 'bert')
        
    Returns:
        Prediction dictionary
    """
    print(f"\n🔍 Analyzing text with {model_type.upper()} model...")
    print(f"📝 Input: {text[:100]}...")
    
    if model_type == 'ml':
        return predict_with_ml(text)
    elif model_type == 'lstm':
        return predict_with_lstm(text)
    elif model_type == 'bert':
        return predict_with_bert(text)
    else:
        raise ValueError("Invalid model_type. Choose from: 'ml', 'lstm', 'bert'")


# Example usage
if __name__ == "__main__":
    # Test texts
    test_texts = [
        "This policy is absolutely terrible and will destroy our country!",
        "The government announced a new healthcare policy today.",
        "Women are naturally better at caring for children than men.",
        "The meeting has been scheduled for 3 PM tomorrow."
    ]
    
    print("="*60)
    print("🧪 BIAS DETECTION PREDICTIONS")
    print("="*60)
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Text: {text}")
        print("-"*60)
        
        # Predict with ML (default)
        result = predict_bias(text, model_type='ml')
        
        print(f"🤖 Model: {result['model']}")
        print(f"📊 Prediction: {result['prediction']}")
        print(f"📈 Confidence: {result['confidence']}%")
        
        if 'key_biased_words' in result and result['key_biased_words']:
            print(f"⚠️  Biased Words Found: {[w[0] for w in result['key_biased_words']]}")