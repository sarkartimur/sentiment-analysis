import torch
import numpy as np
from typing import List, Optional
from constants import BERT_MAX_TOKENS, BERT_MODEL
from model.protocols import Classifier, ModelContainer, ModelSettings
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
from data_loader import DataLoader
from datasets import DatasetDict, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class BERTClassifier(Classifier):

    data: DatasetDict
    model: BertForSequenceClassification

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def fit(self, X_train, y_train):
        self.data = self.loader.load_data_dict()
        self.__model_init()
        
        def tokenize_function(examples):
            return self.tokenizer(examples["features"],
                            padding="max_length",
                            truncation=True,
                            max_length=512)

        tokenized_datasets = self.data.map(tokenize_function, batched=True)
        tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            logging_dir="./logs",
            logging_steps=100,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        # Param search
        # def optuna_hp_space(trial):
        #     return {
        #         "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        #         "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
        #         "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
        #     }

        # trainer = Trainer(
        #     model=None,  # Important: set to None when using model_init
        #     args=training_args,
        #     train_dataset=tokenized_datasets["train"],
        #     eval_dataset=tokenized_datasets["test"],
        #     compute_metrics=compute_metrics,
        #     model_init=model_init
        # )

        # best_run = trainer.hyperparameter_search(
        #     direction="minimize", # Minimize eval_loss
        #     hp_space=optuna_hp_space,
        #     n_trials=20
        # )
        # print(best_run)

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            
            precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(labels, predictions, average=None)
            metrics = {
                "accuracy": accuracy,
                "weighted_precision": precision,
                "weighted_recall": recall,
                "weighted_f1": f1,
                "class_0_precision": precision_class[0],
                "class_0_recall": recall_class[0],
                "class_0_f1": f1_class[0],
                "class_1_precision": precision_class[1],
                "class_1_recall": recall_class[1],
                "class_1_f1": f1_class[1],
            }
            
            return metrics

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=compute_metrics
        )

        trainer.train()

        results = trainer.evaluate()
        print(f"Final evaluation results: {results}")

        trainer.save_model("./imdb_sentiment_reduced")
        self.tokenizer.save_pretrained("./imdb_sentiment_reduced")
        return self

    def predict_proba(self, X):
        self.model.eval()
        self.model.to(self.device)
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
    

    def __model_init(self):
        self.model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL,
            num_labels=2
        )
        
        # Freeze all base BERT parameters
        for param in self.model.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last n layers
        n = 2
        for i in range(-n, 0):
            for param in self.model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        # Unfreeze the classifier layer
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)


class BERTModelContainer(ModelContainer):
    def __init__(self, loader: DataLoader, model: BERTClassifier, settings = ModelSettings()):
        self.loader = loader
        self.model = model
        self.settings = settings


    def train(self) -> None:
        self.data = self.loader.load_data()
        self.model.fit(self.data.X_train, self.data.y_train)

    def test(self) -> None:
        if (self.data is None):
            self.data = self.loader.load_data()

        y_pred_proba = self.model.predict_proba(self.data.X_test)[:, 1]
        y_pred = (y_pred_proba >= self.settings.threshold).astype(int)

        self._output_metrics(y_pred, y_pred_proba)


    def predict_list(self, X: List[str]) -> np.ndarray:
        return self.model.predict_proba(X)