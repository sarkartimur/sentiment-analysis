import numpy as np
import matplotlib.pyplot as plt
from bert_container import BERTContainer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from lime.lime_text import LimeTextExplainer
import util
from util import RANDOM_SEED
import time
import pickle


SAMPLE_SIZE = 2000
TEST_RATIO = 0.2
# Note: small values will lead to "bag of words problem"
# e.g. "This movie is not bad" will be classified as negative
# with high confidence, but longer, more nuanced text will be classified correctly
EMBEDDING_DIM = 150
POOLING_STRATEGY='mean'


# Fix for non-deterministic cv/test accuracy
np.random.seed(RANDOM_SEED)

train_texts, train_labels, test_texts, test_labels = util.load_data(sample_size=SAMPLE_SIZE, test_ratio=TEST_RATIO, imbalance_ratio=1.0)

bert = BERTContainer()
# bert = BERTContainer('bert-base-multilingual-cased')

print("\nExtracting BERT embeddings for training data...")
start_time = time.time()
train_embeddings = bert.get_bert_embeddings(texts=train_texts.tolist(), pooling_strategy=POOLING_STRATEGY)
# train_embeddings = bert.get_gradient_embeddings(texts=train_texts.iloc[:, 0].tolist())
print(f"Embedding extraction time: {time.time() - start_time:.2f} seconds")

print("\nExtracting BERT embeddings for test data...")
test_embeddings = bert.get_bert_embeddings(texts=test_texts.tolist(), pooling_strategy=POOLING_STRATEGY)
# test_embeddings = bert.get_gradient_embeddings(texts=test_texts.iloc[:, 0].tolist())

print(f"\nReducing dimensions to {EMBEDDING_DIM} using PCA...")
train_embeddings_reduced, pca_reducer = util.reduce_dimensions(
    train_embeddings, n_components=EMBEDDING_DIM, method='pca'
)
test_embeddings_reduced = pca_reducer.transform(test_embeddings)

print(f"Train set shape: {train_embeddings_reduced.shape}")

# train_set = np.hstack((train_embeddings_reduced, bert.enhance_embeddings(train_embeddings)))
# test_set = np.hstack((test_embeddings_reduced, bert.enhance_embeddings(test_embeddings)))

model = util.train_svc(train_embeddings_reduced, train_labels)
y_pred = model.predict(test_embeddings_reduced)

ConfusionMatrixDisplay.from_predictions(test_labels, y_pred, normalize='all')
accuracy = accuracy_score(test_labels, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(test_labels, y_pred))

incorrect_idices = util.analyze_errors(test_labels.values, y_pred, test_texts.values)


def lime_explain(txt):
    explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
    exp = explainer.explain_instance(
        txt, 
        predict_arr,
        num_features=20,
        num_samples=3000
    )
    # exp.as_pyplot_figure()
    # plt.show()
    exp.show_in_notebook()

def predict_arr(texts):
    e = bert.get_bert_embeddings(texts=texts, pooling_strategy=POOLING_STRATEGY)
    er = pca_reducer.transform(e)
    return model.predict_proba(er)

def predict(text):
    return predict_arr([text])

def save(filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f, protocol=5)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)