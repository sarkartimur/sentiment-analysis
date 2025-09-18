import numpy as np
from bert_container import BERTContainer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc, roc_auc_score, ConfusionMatrixDisplay
import util
from util import RANDOM_SEED
import time


SAMPLE_SIZE = 2000
TEST_RATIO = 0.2
# Note: small values will lead to "bag of words problem"
# e.g. "This movie is not bad" will be classified as negative
# with high confidence, but longer, more nuanced text will be classified correctly
EMBEDDING_DIM = 150
DIM_REDUCTION = 'pca'
POOLING_STRATEGY='mean'


# Fix for non-deterministic cv/test accuracy
np.random.seed(RANDOM_SEED)

train_texts, train_labels, test_texts, test_labels = util.load_data(sample_size=SAMPLE_SIZE, test_ratio=TEST_RATIO, imbalance_ratio=1.0)

bert = BERTContainer()
# bert = BERTContainer('bert-base-multilingual-cased')

print("\nExtracting BERT embeddings for training data...")
start_time = time.time()
train_embeddings = bert.get_bert_embeddings(texts=train_texts.tolist(), pooling_strategy=POOLING_STRATEGY)
print(f"Embedding extraction time: {time.time() - start_time:.2f} seconds")

print("\nExtracting BERT embeddings for test data...")
test_embeddings = bert.get_bert_embeddings(texts=test_texts.tolist(), pooling_strategy=POOLING_STRATEGY)

print(f"\nReducing dimensions to {EMBEDDING_DIM} using {DIM_REDUCTION}...")
train_embeddings_reduced, pca_reducer = util.reduce_dimensions(
    train_embeddings, n_components=EMBEDDING_DIM, method=DIM_REDUCTION
)
test_embeddings_reduced = pca_reducer.transform(test_embeddings)

print(f"Train set shape: {train_embeddings_reduced.shape}")
print(f"Test set shape: {test_embeddings_reduced.shape}")

model = util.train_svc(train_embeddings_reduced, train_labels)
y_pred = model.predict(test_embeddings_reduced)
y_proba = model.predict_proba(test_embeddings_reduced)

def compute_metrics():
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(test_labels, y_pred))

    roc_auc = roc_auc_score(test_labels, y_proba[:, 1])
    print(f"ROC-AUC: {roc_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(test_labels, y_proba[:, 1])
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")

    util.calculate_certainties(y_proba, test_labels)

    ConfusionMatrixDisplay.from_predictions(test_labels, y_pred, normalize='all')

compute_metrics()

incorrect_idices = util.analyze_errors(test_labels.values, y_pred, test_texts.values)


def predict_arr(texts):
    e = bert.get_bert_embeddings(texts=texts, pooling_strategy=POOLING_STRATEGY)
    er = pca_reducer.transform(e)
    return model.predict_proba(er)

def predict(text):
    return predict_arr([text])