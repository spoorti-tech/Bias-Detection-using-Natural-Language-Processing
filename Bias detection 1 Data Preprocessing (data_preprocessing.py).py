
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Extended stopwords for bias detection (common neutral words)
custom_stopwords = {'said', 'says', 'say', 'would', 'could', 'also', 'one', 'two', 
                    'first', 'new', 'like', 'just', 'even', 'get', 'got', 'go', 'went'}
stop_words.update(custom_stopwords)


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for bias detection.
    
    This class handles all text cleaning operations including:
    - Lowercasing
    - URL, mention, hashtag removal
    - Emoji removal
    - Special character removal
    - Stopword removal
    - Lemmatization
    """
    
    def __init__(self, use_spacy=True):
        """
        Initialize the text preprocessor.
        
        Args:
            use_spacy: Whether to use spaCy for lemmatization (more accurate)
        """
        self.use_spacy = use_spacy
    
    def clean_text(self, text):
        """
        Basic text cleaning: lowercase, remove URLs, mentions, special chars.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from tokenized text.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered list of tokens
        """
        return [word for word in tokens if word not in stop_words and len(word) > 2]
    
    def lemmatize_text(self, tokens):
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        if self.use_spacy:
            # Use spaCy for better lemmatization
            doc = nlp(' '.join(tokens))
            return [token.lemma_ for token in doc]
        else:
            # Use NLTK WordNet lemmatizer
            return [lemmatizer.lemmatize(word) for word in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        # Step 1: Clean text
        cleaned = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = word_tokenize(cleaned)
        
        # Step 3: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 4: Lemmatize
        tokens = self.lemmatize_text(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts):
        """
        Process a batch of texts efficiently.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of preprocessed text strings
        """
        return [self.preprocess(text) for text in texts]


def load_and_preprocess_data(filepath, text_col='text', label_col='label'):
    """
    Load dataset and apply preprocessing.
    
    Args:
        filepath: Path to CSV file
        text_col: Name of text column
        label_col: Name of label column
    
    Returns:
        Processed DataFrame with original and processed text
    """
    print(f"📂 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Label distribution:\n{df[label_col].value_counts()}")
    
    # Check for class imbalance
    class_counts = df[label_col].value_counts()
    if class_counts.min() / class_counts.max() < 0.5:
        print("⚠️ Warning: Class imbalance detected!")
        print(f"Ratio: {class_counts.min() / class_counts.max():.2f}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    print("🔄 Preprocessing texts (this may take a while)...")
    df['processed_text'] = df[text_col].apply(preprocessor.preprocess)
    
    # Remove empty texts
    df = df[df['processed_text'].str.len() > 10]
    
    print(f"✅ Preprocessing complete. Final shape: {df.shape}")
    
    return df


def create_train_test_split(df, text_col='processed_text', label_col='label', 
                           test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    
    Args:
        df: DataFrame with text and labels
        text_col: Name of text column
        label_col: Name of label column
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X = df[text_col]
    y = df[label_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Train set size: {len(X_train)}")
    print(f"📊 Test set size: {len(X_test)}")
    print(f"Train label distribution:\n{y_train.value_counts()}")
    print(f"Test label distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def balance_classes(X, y, random_state=42):
    """
    Balance classes using random undersampling.
    
    Args:
        X: Features (texts)
        y: Labels
        
    Returns:
        Balanced X and y
    """
    from sklearn.utils import resample
    
    # Combine X and y
    data = pd.DataFrame({'text': X, 'label': y})
    
    # Separate majority and minority classes
    df_majority = data[data['label'] == 0]
    df_minority = data[data['label'] == 1]
    
    # Unsample majority class
    n_samples = len(df_minority)
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,
                                       n_samples=n_samples,
                                       random_state=random_state)
    
    # Combine
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state)
    
    print(f"✅ Balanced dataset: {len(df_balanced)} samples")
    print(f"Label distribution:\n{df_balanced['label'].value_counts()}")
    
    return df_balanced['text'], df_balanced['label']