import numpy as np
from sentiment_classifier import BERTSentimentClassifier, load_imdb_data
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import time
from datasets import load_dataset

# Configuration - you can modify these values
SAMPLE_SIZE = 2000
EMBEDDING_DIM = 50
RANDOM_STATE = 42

print("Starting BERT + XGBoost Sentiment Analysis...")

# Load data
train_texts, train_labels, test_texts, test_labels = load_imdb_data(SAMPLE_SIZE)

# # Initialize BERT classifier
classifier = BERTSentimentClassifier()

# Extract BERT embeddings
print("\nExtracting BERT embeddings for training data...")
start_time = time.time()
train_embeddings = classifier.get_bert_embeddings(train_texts)
print(f"Embedding extraction time: {time.time() - start_time:.2f} seconds")

print("\nExtracting BERT embeddings for test data...")
test_embeddings = classifier.get_bert_embeddings(test_texts)

# Reduce dimensions
print(f"\nReducing dimensions to {EMBEDDING_DIM} using PCA...")
train_embeddings_reduced, pca_reducer = classifier.reduce_dimensions(
    train_embeddings, n_components=EMBEDDING_DIM, method='pca'
)
test_embeddings_reduced = pca_reducer.transform(test_embeddings)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_embeddings_reduced, train_labels, 
    test_size=0.2, random_state=RANDOM_STATE, stratify=train_labels
)

# Train XGBoost model
print("\nTraining XGBoost model...")
xgb_model = classifier.train_xgboost(X_train, y_train, X_val, y_val)

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred, y_pred_proba = classifier.evaluate_model(xgb_model, test_embeddings_reduced, test_labels)

# Feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(xgb_model, max_num_features=20)
plt.title('Top 20 Feature Importances')
plt.show()

# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()