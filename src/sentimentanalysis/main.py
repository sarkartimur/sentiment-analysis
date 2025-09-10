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
TEST_RATIO = 0.2
EMBEDDING_DIM = 50
POOLING_STRATEGY='mean'


train_texts, train_labels, test_texts, test_labels = util.load_data(sample_size=SAMPLE_SIZE, test_ratio=TEST_RATIO)

bert = BERTContainer()
# bert = BERTContainer('bert-base-multilingual-cased')

print("\nExtracting BERT embeddings for training data...")
start_time = time.time()
train_embeddings = bert.get_bert_embeddings(texts=train_texts.iloc[:, 0].tolist(), pooling_strategy=POOLING_STRATEGY)
# train_embeddings = bert.get_gradient_embeddings(texts=train_texts.iloc[:, 0].tolist())
print(f"Embedding extraction time: {time.time() - start_time:.2f} seconds")

print("\nExtracting BERT embeddings for test data...")
test_embeddings = bert.get_bert_embeddings(texts=test_texts.iloc[:, 0].tolist(), pooling_strategy=POOLING_STRATEGY)
# test_embeddings = bert.get_gradient_embeddings(texts=test_texts.iloc[:, 0].tolist())

print(f"\nReducing dimensions to {EMBEDDING_DIM} using PCA...")
train_embeddings_reduced, pca_reducer = util.reduce_dimensions(
    train_embeddings, n_components=EMBEDDING_DIM, method='pca'
)
test_embeddings_reduced = pca_reducer.transform(test_embeddings)

# train_set = np.hstack((train_embeddings_reduced, bert.enhance_embeddings(train_embeddings)))
# test_set = np.hstack((test_embeddings_reduced, bert.enhance_embeddings(test_embeddings)))

model = util.train_logistic_regression(train_embeddings_reduced, train_labels)
y_pred = model.predict(test_embeddings_reduced)

ConfusionMatrixDisplay.from_predictions(test_labels, y_pred, normalize='all')
accuracy = accuracy_score(test_labels, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(test_labels, y_pred))

incorrect_idices = util.analyze_errors(test_labels.values, y_pred, test_texts.values)


def predict(text):
    e = bert.get_bert_embeddings(texts=[text], pooling_strategy=POOLING_STRATEGY)
    er = pca_reducer.transform(e)
    return model.predict_proba(er)