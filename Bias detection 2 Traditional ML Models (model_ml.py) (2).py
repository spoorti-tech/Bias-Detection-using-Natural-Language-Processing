"""
Traditional Machine Learning Models for Bias Detection
Uses TF-IDF vectorization with Logistic Regression, Naive Bayes, and SVM.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import GridSearchCV
import pickle
import warnings
warnings.filterwarnings('ignore')


class MLModels:
    """
    Collection of ML models for text classification.
    
    This class provides:
    - TF-IDF vectorization
    - Logistic Regression
    - Naive Bayes
    - Support Vector Machine
    - Hyperparameter tuning
    - Model evaluation
    - Feature importance extraction
    """
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.results = {}
    
    def create_vectorizer(self, max_features=5000, ngram_range=(1, 2)):
        """
        Create TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams (1,2) for unigrams and bigrams
        
        Returns:
            Fitted TfidfVectorizer
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True  # Apply log scaling
        )
        return self.vectorizer
    
    def fit_vectorizer(self, X_train):
        """
        Fit vectorizer on training data.
        
        Args:
            X_train: Training texts
            
        Returns:
            Transformed training data
        """
        if self.vectorizer is None:
            self.create_vectorizer()
        self.X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"✅ Vectorizer fitted. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self.X_train_tfidf
    
    def transform_data(self, X):
        """
        Transform data using fitted vectorizer.
        
        Args:
            X: Texts to transform
            
        Returns:
            Transformed texts
        """
        return self.vectorizer.transform(X)
    
    def train_logistic_regression(self, X_train, y_train, use_grid_search=True):
        """
        Train Logistic Regression model.
        
        Hyperparameters:
        - C: Regularization strength (inverse)
        - solver: Optimization algorithm
        """
        print("🚀 Training Logistic Regression...")
        
        if use_grid_search:
            param_grid = {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [1000]
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(random_state=42, class_weight='balanced'),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.models['logistic_regression'] = grid_search.best_estimator_
            print(f"✅ Best params: {grid_search.best_params_}")
        else:
            self.models['logistic_regression'] = LogisticRegression(
                C=1, solver='lbfgs', max_iter=1000, 
                random_state=42, class_weight='balanced'
            )
            self.models['logistic_regression'].fit(X_train, y_train)
    
    def train_naive_bayes(self, X_train, y_train):
        """
        Train Multinomial Naive Bayes model.
        
        Hyperparameters:
        - alpha: Smoothing parameter
        """
        print("🚀 Training Naive Bayes...")
        
        param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
        
        grid_search = GridSearchCV(
            MultinomialNB(),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['naive_bayes'] = grid_search.best_estimator_
        print(f"✅ Best alpha: {grid_search.best_params_}")
    
    def train_svm(self, X_train, y_train, use_grid_search=True):
        """
        Train Support Vector Machine model.
        
        Hyperparameters:
        - C: Regularization parameter
        - kernel: Kernel type (linear, rbf)
        """
        print("🚀 Training SVM...")
        
        if use_grid_search:
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            
            grid_search = GridSearchCV(
                SVC(random_state=42, class_weight='balanced', probability=True),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.models['svm'] = grid_search.best_estimator_
            print(f"✅ Best params: {grid_search.best_params_}")
        else:
            self.models['svm'] = SVC(
                C=1, kernel='linear', random_state=42, 
                class_weight='balanced', probability=True
            )
            self.models['svm'].fit(X_train, y_train)
    
    def train_all(self, X_train, y_train):
        """
        Train all ML models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.train_logistic_regression(X_train, y_train)
        self.train_naive_bayes(X_train, y_train)
        self.train_svm(X_train, y_train)
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        self.results[model_name] = metrics
        
        print(f"\n{'='*50}")
        print(f"📊 {model_name.upper()} Results")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Biased', 'Biased']))
        
        return metrics
    
    def evaluate_all(self, X_test, y_test):
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        for model_name in self.models:
            self.evaluate_model(model_name, X_test, y_test)
    
    def get_feature_importance(self, model_name, top_n=20):
        """
        Get top biased words based on TF-IDF weights.
        
        Args:
            model_name: Name of the model
            top_n: Number of top words to return
            
        Returns:
            Dictionary with biased and non-biased words
        """
        if model_name not in self.models:
            print("Model not found!")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'coef_'):
            # Logistic Regression
            importance = model.coef_[0]
        elif hasattr(model, 'feature_log_prob_'):
            # Naive Bayes - calculate difference between classes
            importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        else:
            print("Model doesn't support feature importance extraction")
            return None
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top features for biased class (positive values)
        top_positive_idx = np.argsort(importance)[-top_n:][::-1]
        top_negative_idx = np.argsort(importance)[:top_n]
        
        return {
            'biased_words': [(feature_names[i], importance[i]) for i in top_positive_idx],
            'non_biased_words': [(feature_names[i], importance[i]) for i in top_negative_idx]
        }
    
    def save_models(self, prefix='ml'):
        """
        Save models and vectorizer to disk.
        
        Args:
            prefix: Prefix for saved files
        """
        import joblib
        
        for model_name, model in self.models.items():
            joblib.dump(model, f'{prefix}_{model_name}.pkl')
        
        joblib.dump(self.vectorizer, f'{prefix}_vectorizer.pkl')
        print(f"✅ Models saved with prefix: {prefix}")
    
    def load_models(self, prefix='ml'):
        """
        Load models from disk.
        
        Args:
            prefix: Prefix for saved files
        """
        import joblib
        
        self.vectorizer = joblib.load(f'{prefix}_vectorizer.pkl')
        
        for model_name in ['logistic_regression', 'naive_bayes', 'svm']:
            try:
                self.models[model_name] = joblib.load(f'{prefix}_{model_name}.pkl')
            except FileNotFoundError:
                print(f"Warning: {model_name} not found")
        
        print("✅ Models loaded successfully")