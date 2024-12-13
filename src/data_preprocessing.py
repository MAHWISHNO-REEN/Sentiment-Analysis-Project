# src/data_preprocessing.py
import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to load data
# def load_data(base_path):
#     texts = []
#     labels = []
#     for subset in ['train', 'test']:
#         for label_type in ['pos', 'neg']:
#             folder_path = os.path.join(base_path, subset, label_type)
#             label = 1 if label_type == 'pos' else 0
#             for file_name in os.listdir(folder_path):
#                 with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
#                     texts.append(f.read())
#                     labels.append(label)
#     return pd.DataFrame({'text': texts, 'label': labels})

def load_data(base_path):
    texts = []
    labels = []
    for subset in ['train', 'test']:
        for label_type in ['pos', 'neg']:
            folder_path = os.path.join(base_path, subset, label_type)
            label = 1 if label_type == 'pos' else 0
            # Modify the loop here
            for i, file_name in enumerate(os.listdir(folder_path)):
                if i > 100:  # Limit to the first 100 files for testing
                    break
                print("Reading:", file_name)  # Optional: Print to track progress
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
    return texts, labels


# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Function to remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Function to apply lemmatization
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

# Tokenize and preprocess the data
def preprocess_data(df):
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['tokens'] = df['cleaned_text'].apply(word_tokenize)
    df['filtered_tokens'] = df['tokens'].apply(remove_stopwords)
    df['lemmatized_tokens'] = df['filtered_tokens'].apply(lemmatize_tokens)
    return df
