import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from lime.lime_text import LimeTextExplainer


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

def lime_explain(txt):
    explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
    exp = explainer.explain_instance(
        txt, 
        predict_sentiment,
        num_features=10,
        num_samples=3000
    )
    # exp.as_pyplot_figure()
    # plt.show()
    exp.show_in_notebook()