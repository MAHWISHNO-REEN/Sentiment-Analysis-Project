# src/model_training.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF vectorization and data split
def vectorize_data(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()
    y = df['label'].values
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer

# Train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='logistic_regression'):
    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'svm':
        model = SVC()
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)
