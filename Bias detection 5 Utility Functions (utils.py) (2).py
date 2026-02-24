"""
Utility functions for visualization and evaluation.
Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from wordcloud import WordCloud
import os


def plot_training_history(history, model_name):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras training history object
        model_name: Name of the model for saving plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='o', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', marker='o', linewidth=2)
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Val Loss', marker='o', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Saved training history plot: {model_name}_history.png")


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Custom save path (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Biased', 'Biased'],
                yticklabels=['Non-Biased', 'Biased'],
                annot_kws={'size': 14})
    plt