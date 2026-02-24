"""
LSTM-based Deep Learning Model for Bias Detection
Uses word embeddings and bidirectional LSTM layers.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LSTM, Embedding, Dropout, 
                                     Bidirectional, Input, SpatialDropout1D,
                                     GlobalMaxPooling1D, GlobalAveragePooling1D, 
                                     Concatenate)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                       ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class LSTMBasedClassifier:
    """
    LSTM-based text classifier for bias detection.
    
    Architecture:
    - Embedding Layer (converts words to vectors)
    - Spatial Dropout (regularization)
    - Bidirectional LSTM (captures context in both directions)
    - Global Max/Average Pooling (extracts key features)
    - Dense layers with Dropout
    - Sigmoid output (binary classification)
    """
    
    def __init__(self, max_words=10000, max_len=200, embedding_dim=128):
        """
        Initialize LSTM model parameters.
        
        Args:
            max_words: Maximum vocabulary size
            max_len: Maximum sequence length for padding
            embedding_dim: Dimension of word embeddings
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.history = None
    
    def prepare_data(self, X_train, X_test):
        """
        Tokenize and pad sequences for LSTM.
        
        Args:
            X_train: Training texts
            X_test: Test texts
            
        Returns:
            Padded training and test sequences
        """
        print("🔄 Preparing data for LSTM...")
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        
        # Convert texts to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences to ensure uniform length
        self.X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len, 
                                         padding='post', truncating='post')
        self.X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len, 
                                        padding='post', truncating='post')
        
        print(f"✅ Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"✅ Padded sequence shape: {self.X_train_pad.shape}")
        
        return self.X_train_pad, self.X_test_pad
    
    def build_bilstm_model(self, dropout_rate=0.3):
        """
        Build Bidirectional LSTM model.
        
        Architecture:
        - Embedding Layer: Converts word indices to dense vectors
        - Spatial Dropout: Regularization (drops entire feature maps)
        - Bidirectional LSTM x2: Captures forward and backward context
        - Global Max + Average Pooling: Captures most important features
        - Concatenate: Combines pooling results
        - Dense Layers: Final classification head
        - Sigmoid: Binary output
        """
        print("🏗️ Building BiLSTM Model...")
        
        self.model = Sequential([
            # Embedding Layer
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            
            # Spatial Dropout (drop entire 1D feature maps)
            SpatialDropout1D(0.2),
            
            # First Bidirectional LSTM
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)),
            
            # Second Bidirectional LSTM
            Bidirectional(LSTM(32, return_sequences=False, dropout=0.1)),
            
            # Concatenate Global Max and Average Pooling
            GlobalMaxPooling1D(),
            GlobalAveragePooling1D(),
            Concatenate(),
            
            # Dense Layers
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            
            # Output Layer (Binary Classification)
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print("Model Summary:")
        self.model.summary()
        return self.model
    
    def train(self, y_train, y_val, epochs=20, batch_size=32, 
              learning_rate=0.001, dropout_rate=0.3):
        """
        Train the LSTM model.
        
        Hyperparameters:
        - epochs: Number of training epochs (20-30 recommended)
        - batch_size: Number of samples per batch (32-64 recommended)
        - learning_rate: Adam optimizer learning rate
        - dropout_rate: Dropout rate for regularization
        
        Args:
            y_train: Training labels
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            dropout_rate: Dropout rate
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_bilstm_model(dropout_rate)
        
        # Recompile with custom learning rate if needed
        if learning_rate != 0.001:
            self.model.optimizer = Adam(learning_rate=learning_rate)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            ModelCheckpoint('best_lstm_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        print(f"\n🚀 Training LSTM for {epochs} epochs...")
        
        self.history = self.model.fit(
            self.X_train_pad, y_train,
            validation_data=(self.X_test_pad, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, y_test):
        """
        Evaluate model on test data.
        
        Args:
            y_test: Test labels
            
        Returns:
            Predictions and probabilities
        """
        print("\n📊 Evaluating LSTM Model...")
        
        y_pred_prob = self.model.predict(self.X_test_pad)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Biased', 'Biased']))
        
        return y_pred, y_pred_prob
    
    def predict(self, text):
        """
        Predict bias for a single text.
        
        Args:
            text: Input text string
        
        Returns:
            Dictionary with prediction and probability
        """
        # Preprocess text
        seq = self.tokenizer.texts_to_sequences([text])
        pad_seq = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
        
        # Predict
        prob = self.model.predict(pad_seq, verbose=0)[0][0]
        
        result = {
            'prediction': 'Biased' if prob > 0.5 else 'Non-Biased',
            'confidence': round(prob * 100, 2) if prob > 0.5 else round((1 - prob) * 100, 2),
            'probability_bias': prob
        }
        
        return result
    
    def save_model(self, filepath='lstm_bias_model.h5'):
        """Save LSTM model and tokenizer."""
        self.model.save(filepath)
        
        # Save tokenizer
        import pickle
        with open('lstm_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='lstm_bias_model.h5'):
        """Load LSTM model and tokenizer."""
        self.model = tf.keras.models.load_model(filepath)
        
        import pickle
        with open('lstm_tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print(f"✅ Model loaded from {filepath}")