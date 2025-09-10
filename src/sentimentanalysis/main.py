import numpy as np
from bert_container import BERTContainer
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import util
from util import RANDOM_SEED
import time


SAMPLE_SIZE = 2000
EMBEDDING_DIM = 50
POOLING_STRATEGY='mean'


print("Starting BERT + XGBoost Sentiment Analysis...")

train_texts, train_labels, test_texts, test_labels = util.load_data(sample_size=SAMPLE_SIZE, test_ratio=0.2)

classifier = BERTContainer()

print("\nExtracting BERT embeddings for training data...")
start_time = time.time()
train_embeddings = classifier.get_bert_embeddings(texts=train_texts.iloc[:, 0].tolist(), pooling_strategy=POOLING_STRATEGY)
print(f"Embedding extraction time: {time.time() - start_time:.2f} seconds")

print("\nExtracting BERT embeddings for test data...")
test_embeddings = classifier.get_bert_embeddings(texts=test_texts.iloc[:, 0].tolist(), pooling_strategy=POOLING_STRATEGY)

print(f"\nReducing dimensions to {EMBEDDING_DIM} using PCA...")
train_embeddings_reduced, pca_reducer = util.reduce_dimensions(
    train_embeddings, n_components=EMBEDDING_DIM, method='pca'
)
test_embeddings_reduced = pca_reducer.transform(test_embeddings)

X_train, X_val, y_train, y_val = train_test_split(
    train_embeddings_reduced, train_labels, 
    test_size=0.2, random_state=RANDOM_SEED, stratify=train_labels
)

print("\nTraining XGBoost model...")
xgb_model = util.train_xgboost(X_train, y_train, X_val, y_val)

y_pred = xgb_model.predict(test_embeddings_reduced)

ConfusionMatrixDisplay.from_predictions(test_labels, y_pred, normalize='all')

accuracy = accuracy_score(test_labels, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(test_labels, y_pred))

# Feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(xgb_model, max_num_features=20)
plt.title('Top 20 Feature Importances')
plt.show()