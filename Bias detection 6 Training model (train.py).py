"""
Main Training Script for Bias Detection
Trains ML, LSTM, and BERT models.

Author: AI Assistant
Date: 2024
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Import our custom modules
from data_preprocessing import load_and_preprocess_data, create_train_test_split
from model_ml import MLModels
from model_lstm import LSTMBasedClassifier
from model_bert import BERTClassifier
from utils import plot_training_history, plot_confusion_matrix, plot_roc_curve, plot_wordcloud

# Check for GPU
print("="*60)
print("🖥️  SYSTEM CHECK")
print("="*60)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {len(gpus)} device(s)")
    for gpu in gpus:
        print(f"   - {gpu}")
else:
    print("⚠️  No GPU found, using CPU (training will be slower)")

# Configuration
DATA_PATH = 'dataset/bias_data.csv'
MODEL_TO_TRAIN = 'ml'  # Options: 'ml', 'lstm', 'bert', 'all'


def train_ml_models(X_train, X_test, y_train, y_test):
    """Train Traditional ML Models"""
    print("\n" + "="*60)
    print("📚 TRAINING TRADITIONAL ML MODELS")
    print("="*60)
    
    ml = MLModels()
    
    # TF-IDF Vectorization
    print("🔄 Creating TF-IDF Features...")
    ml.fit_vectorizer(X_train)
    X_train_tfidf = ml.transform_data(X_train)
    X_test_tfidf = ml.transform_data(X_test)
    
    # Train all models
    ml.train_all(X_train_tfidf, y_train)
    
    # Evaluate all
    ml.evaluate_all(X_test_tfidf, y_test)
    
    # Get feature importance for Logistic Regression
    print("\n📊 Top Biased Words (Logistic Regression):")
    importance = ml.get_feature_importance('logistic_regression', top_n=15)
    if importance:
        print("Biased Words:", [w[0] for w in importance['biased_words'][:10]])
        print("Non-Biased Words:", [w[0] for w in importance['non_biased_words'][:10]])
    
    # Save models
    ml.save_models('ml')
    
    return ml


def train_lstm_model(X_train, X_test, y_train, y_test):
    """Train LSTM Model"""
    print("\n" + "="*60)
    print("🧠 TRAINING LSTM MODEL")
    print("="*60)
    
    lstm = LSTMBasedClassifier(max_words=10000, max_len=150, embedding_dim=128)
    
    # Prepare data
    X_train_pad, X_test_pad = lstm.prepare_data(X_train, X_test)
    
    # Build model
    lstm.build_bilstm_model(dropout_rate=0.3)
    
    # Train
    history = lstm.train(y_train.values, y_test.values, epochs=15, batch_size=32)
    
    # Evaluate
    y_pred, y_prob = lstm.evaluate(y_test.values)
    
    # Visualize
    plot_training_history(history, 'LSTM')
    plot_confusion_matrix(y_test.values, y_pred, 'LSTM')
    plot_roc_curve(y_test.values, y_prob.flatten(), 'LSTM')
    
    # Save
    lstm.save_model('lstm_bias_model.h5')
    
    return lstm


def train_bert_model(X_train, X_test, y_train, y_test):
    """Train BERT Model"""
    print("\n" + "="*60)
    print("🤖 TRAINING BERT MODEL")
    print("="*60)
    
    bert = BERTClassifier(model_name='bert-base-uncased', max_len=128)
    
    # Build model
    bert.build_model(learning_rate=2e-5)
    
    # Train (Note: BERT needs very few epochs)
    history = bert.train(X_train, y_train, X_test, y_test, epochs=3, batch_size=16)
    
    # Predict
    print("\n📊 Evaluating BERT Model...")
    results = bert.predict(X_test.tolist()[:100])  # Evaluate on subset for speed
    
    # Visualize
    plot_training_history(history, 'BERT')
    
    # Save
    bert.save_model('bert_bias_model')
    
    return bert


def visualize_data(df):
    """Visualize dataset characteristics"""
    print("\n📊 Generating Data Visualizations...")
    
    # Word clouds for biased vs non-biased
    biased_texts = df[df['label'] == 1]['processed_text']
    non_biased_texts = df[df['label'] == 0]['processed_text']
    
    plot_wordcloud(biased_texts, 'Biased Text Word Cloud', 'wordcloud_biased.png')
    plot_wordcloud(non_biased_texts, 'Non-Biased Text Word Cloud', 'wordcloud_non_biased.png')


# ======================
# MAIN EXECUTION
# ======================

if __name__ == "__main__":
    print(f"\n🚀 Starting Bias Detection Project...")
    print(f"📁 Dataset: {DATA_PATH}")
    print(f"🎯 Training Mode: {MODEL_TO_TRAIN.upper()}")
    
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data(DATA_PATH, text_col='text', label_col='label')
    
    # Step 2: Visualize data
    visualize_data(df)
    
    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = create_train_test_split(
        df, test_size=0.2, random_state=42
    )
    
    # Step 4: Train selected model(s)
    if MODEL_TO_TRAIN in ['ml', 'all']:
        train_ml_models(X_train, X_test, y_train, y_test)
    
    if MODEL_TO_TRAIN in ['lstm', 'all']:
        train_lstm_model(X_train, X_test, y_train, y_test)
    
    if MODEL_TO_TRAIN in ['bert', 'all']:
        train_bert_model(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)