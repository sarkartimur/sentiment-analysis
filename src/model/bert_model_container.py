import torch
import numpy as np
from typing import List
from constants import BERT_MAX_TOKENS
from model.protocols import Classifier, ModelContainer, ModelSettings
from transformers import BertForSequenceClassification, BertTokenizer
from data_loader import DataLoader


class BERTClassifier(Classifier):
    def __init__(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)

    # todo Implementation from bert_ft
    def fit(self, X_train, y_train):
        return super().fit(X_train, y_train)

    def predict_proba(self, X):
        if not isinstance(X, List):
            X = X.to_list()
        result = []
        for i in range(0, len(X)):
            text = X[i]

            inputs = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=BERT_MAX_TOKENS,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                result.append(probabilities)
            
        return np.concatenate(result, axis=0)


    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class BERTModelContainer(ModelContainer):
    def __init__(self, loader: DataLoader, model: BERTClassifier, settings = ModelSettings()):
        self.loader = loader
        self.model = model
        self.settings = settings


    # todo implement
    def train(self) -> None:
        pass

    def test(self) -> None:
        if (self.data is None):
            self.data = self.loader.load_data()

        y_pred_proba = self.model.predict_proba(self.data.X_test)[:, 1]
        y_pred = (y_pred_proba >= self.settings.threshold).astype(int)

        self._output_metrics(y_pred, y_pred_proba)


    def predict_list(self, X: List[str]) -> np.ndarray:
        return self.model.predict_proba(X)