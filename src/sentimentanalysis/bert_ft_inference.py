import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import util
import metrics


MODEL_PATH = "./imdb_sentiment_reduced"
MODEL = BertForSequenceClassification.from_pretrained(MODEL_PATH)
TOKENIZER = BertTokenizer.from_pretrained(MODEL_PATH)


def predict_sentiment(texts, device="cuda"):
    MODEL.eval()
    MODEL.to(device)
    
    result = []
    for i in range(0, len(texts)):
        text = texts[i]

        inputs = TOKENIZER(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = MODEL(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            result.append(probabilities)
        
    return np.concatenate(result, axis=0)

def predict(txt):
    return predict_sentiment([txt])


X_train, y_train, X_test, y_test = util.load_data(sample_size=2000, test_ratio=0.2, imbalance_ratio=1)

threshold = 0.5
y_pred_proba = predict_sentiment(X_test.tolist())
y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)

metrics.compute_metrics(y_test, y_pred, y_pred_proba)

incorrect_idices = util.analyze_errors(y_test.values, y_pred, X_test.values)

# metrics.plot_roc_auc(y_test, y_pred_proba)

# metrics.plot_threshold_graph(y_test, y_pred_proba)

# metrics.plot_class_overlap_graph(y_test, y_pred_proba)

# metrics.plot_calibration_graph(y_test, y_pred_proba)