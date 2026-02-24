"""
BERT-based Transfer Learning Model for Bias Detection
Fine-tunes pretrained BERT for binary classification.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                       ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


class BERTClassifier:
    """
    BERT-based classifier for bias detection.
    
    This model uses the BERT (Bidirectional Encoder Representations from 
    Transformers) architecture for state-of-the-art text classification.
    
    Workflow:
    1. Load pretrained BERT (bert-base-uncased)
    2. Add custom classification head
    3. Fine-tune on our bias dataset
    """
    
    def __init__(self, model_name='bert-base-uncased', max_len=128):
        """
        Initialize BERT model.
        
        Args:
            model_name: Pretrained BERT model name
            max_len: Maximum sequence length
        """
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = None
        self.model = None
        self.history = None
    
    def prepare_tokenizer(self):
        """Load BERT tokenizer."""
        print(f"🔄 Loading {self.model_name} tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        return self.tokenizer
    
    def prepare_data(self, texts, labels=None):
        """
        Tokenize texts for BERT.
        
        Args:
            texts: List of text strings
            labels: List of labels (optional)
        
        Returns:
            Dictionary of input tensors
        """
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors='tf'
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        if labels is not None:
            return input_ids, attention_mask, np.array(labels)
        else:
            return input_ids, attention_mask
    
    def build_model(self, learning_rate=2e-5):
        """
        Build BERT model for sequence classification.
        
        Hyperparameters:
        - learning_rate: Typically 2e-5, 3e-5, 5e-5 for fine-tuning BERT
                        Lower learning rates are crucial for BERT fine-tuning
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            Compiled BERT model
        """
        print("🏗️ Building BERT Model...")
        
        # Load pretrained BERT for sequence classification
        self.model = TFBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # Optional: Freeze BERT layers initially for faster training
        # Uncomment to freeze:
        # for layer in self.model.bert.layers:
        #     layer.trainable = False
        
        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        print("✅ BERT Model built successfully!")
        print("\nModel Architecture:")
        self.model.summary()
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=3, batch_size=16):
        """
        Train BERT model.
        
        Note: BERT typically requires very few epochs (2-4) for fine-tuning.
        Larger epochs may lead to overfitting.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            epochs: Number of epochs (keep low for BERT)
            batch_size: Batch size (keep small due to memory)
            
        Returns:
            Training history
        """
        if self.tokenizer is None:
            self.prepare_tokenizer()
        
        # Prepare data
        print("🔄 Tokenizing datasets (this may take time)...")
        train_ids, train_mask, y_train = self.prepare_data(X_train.tolist(), y_train.tolist())
        val_ids, val_mask, y_val = self.prepare_data(X_val.tolist(), y_val.tolist())
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1),
            ModelCheckpoint('best_bert_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        print(f"\n🚀 Training BERT for {epochs} epochs...")
        
        # Train
        self.history = self.model.fit(
            [train_ids, train_mask],
            y_train,
            validation_data=([val_ids, val_mask], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, texts):
        """
        Predict bias for texts.
        
        Args:
            texts: List of text strings or single text
        
        Returns:
            List of prediction dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        input_ids, attention_mask = self.prepare_data(texts)
        
        # Predict
        predictions = self.model.predict([input_ids, attention_mask], verbose=0)
        logits = predictions.logits
        
        # Convert to probabilities using softmax
        probs = tf.keras.activations.softmax(logits, axis=1).numpy()
        
        # Get class with highest probability
        pred_classes = np.argmax(probs, axis=1)
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text[:50] + '...' if len(text) > 50 else text,
                'prediction': 'Biased' if pred_classes[i] == 1 else 'Non-Biased',
                'confidence': round(np.max(probs[i]) * 100, 2),
                'probabilities': {
                    'Non-Biased': round(probs[i][0] * 100, 2),
                    'Biased': round(probs[i][1] * 100, 2)
                }
            })
        
        return results
    
    def save_model(self, path='bert_bias_model'):
        """Save BERT model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path='bert_bias_model'):
        """Load BERT model and tokenizer."""
        self.model = TFBertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        print(f"✅ Model loaded from {path}")