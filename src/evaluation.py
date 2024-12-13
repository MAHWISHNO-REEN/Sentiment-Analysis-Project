# src/evaluation.py
from sklearn.metrics import classification_report

# Function to evaluate model
def evaluate_model(y_test, y_pred):
    return classification_report(y_test, y_pred)
