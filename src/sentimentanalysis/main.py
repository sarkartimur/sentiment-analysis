import numpy as np
from bert_container import BERTContainer
import util
import metrics
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

X_train, y_train, X_test, y_test = util.load_data(sample_size=SAMPLE_SIZE, test_ratio=TEST_RATIO, imbalance_ratio=1)

bert = BERTContainer()
# bert = BERTContainer('bert-base-multilingual-cased')

print("\nExtracting BERT embeddings for training data...")
start_time = time.time()
train_embeddings = bert.get_bert_embeddings(texts=X_train.tolist(), pooling_strategy=POOLING_STRATEGY)
print(f"Embedding extraction time: {time.time() - start_time:.2f} seconds")

print("\nExtracting BERT embeddings for test data...")
test_embeddings = bert.get_bert_embeddings(texts=X_test.tolist(), pooling_strategy=POOLING_STRATEGY)

print(f"\nReducing dimensions to {EMBEDDING_DIM} using {DIM_REDUCTION}...")
train_embeddings_reduced, pca_reducer = util.reduce_dimensions(
    train_embeddings, n_components=EMBEDDING_DIM, method=DIM_REDUCTION
)
test_embeddings_reduced = pca_reducer.transform(test_embeddings)

print(f"Train set shape: {train_embeddings_reduced.shape}")
print(f"Test set shape: {test_embeddings_reduced.shape}")

model = util.train_svc(train_embeddings_reduced, y_train)
y_pred = model.predict(test_embeddings_reduced)
y_pred_proba = model.predict_proba(test_embeddings_reduced)

# threshold = 0.85
# y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)

metrics.compute_metrics(y_test, y_pred, y_pred_proba)

incorrect_idices = util.analyze_errors(y_test.values, y_pred, X_test.values)

# metrics.plot_roc_auc(y_test, y_pred_proba)

# metrics.plot_threshold_graph(y_test, y_pred_proba)

# metrics.plot_class_overlap_graph(y_test, y_pred_proba)

# metrics.plot_calibration_graph(y_test, y_pred_proba)


# todo make bert singleton, move these to util
def predict_arr(texts):
    e = bert.get_bert_embeddings(texts=texts, pooling_strategy=POOLING_STRATEGY)
    er = pca_reducer.transform(e)
    return model.predict_proba(er)

def predict(text):
    return predict_arr([text])