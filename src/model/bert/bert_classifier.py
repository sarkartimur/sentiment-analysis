from dataclasses import dataclass
import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from model.constants import BERT_MODEL, RANDOM_SEED
from model.protocols import BERTWrapperMixin, ClassifierMixin
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from calibration import get_ece


@dataclass
class BertClassifierSettings:
    num_train_epochs: int = 3
    learning_rate: int = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.05
    dropout_prob: float = 0.25
    # Note: for balanced data use different metric
    metric_for_best_model: str = "eval_class_1_f1"
    greater_is_better: bool = True
    label_smoothing_factor: float = 0.1
    n_layer_unfreeze: int = 3
    temperature_scale: float = 1.0
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: np.ndarray = None
    gradient_accumulation_steps: int = 1
    batch_size: int = 8
    random_seed: int = RANDOM_SEED
    local_model: bool = False
    model_path: str = None
    local_path: str = os.getenv('TEMP') + "\\pretrained\\" + BERT_MODEL.replace('/', '-')
    ouput_dir: str = os.getenv('TEMP') + "\\pretrained\\trainer_output"
    
    def __post_init__(self):
        self.model_path = self.local_path if self.local_model else BERT_MODEL


class BERTClassifier(BERTWrapperMixin, ClassifierMixin):
    _model: BertForSequenceClassification
    __trainer: Optional[Trainer]

    def __init__(self, settings=BertClassifierSettings()):
        super().__init__(model_path=settings.model_path, local_model=settings.local_model)
        self.__settings = settings
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        if self._local_model:
            self._model = BertForSequenceClassification.from_pretrained(self._model_path)
        else:
            self.__model_init()
        print(f"Initialized BERTClassifier with settings: {settings}")


    def train(self, data: DatasetDict):
        if self.__settings.local_model:
            raise ValueError("Model loaded from local path.")

        def tokenize_function(examples):
            return self._tokenizer(examples["features"],
                            padding=True,
                            truncation=True,
                            max_length=self._max_length)

        tokenized_datasets = data.map(tokenize_function, batched=True)
        tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        self.__trainer = self.FocalLossTrainer(
            model=self._model,
            args=self.__build_training_args(),
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=DataCollatorWithPadding(tokenizer=self._tokenizer),
            alpha=(torch.tensor(self.__settings.focal_loss_alpha, dtype=torch.float32)
                        if self.__settings.focal_loss_alpha is not None
                        else None),
            gamma=self.__settings.focal_loss_gamma
        )

        self.__trainer.train()

        return self

    # todo
    def save(self):
        if not self._local_model:
            self.__trainer.save_model(self.__settings.local_path)
            self._tokenizer.save_pretrained(self.__settings.local_path)

    def predict_proba(self, X):
        self._model.eval()
        self._model.to(self._device)
        
        if isinstance(X, pd.DataFrame):
            X = X.to_list()
        elif isinstance(X, (np.ndarray, pd.Series)):
            X = X.tolist()
        
        result = []
        for i in range(0, len(X)):
            text = X[i]

            inputs = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                
                # Apply temperature scaling to the logits
                logits = logits / self.__settings.temperature_scale
                
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                result.append(probabilities)
            
        return np.concatenate(result, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    

    # todo
    def __param_search(self, tokenized_datasets, compute_metrics):
        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
                "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
            }

        self.__trainer = Trainer(
            model=self._model,
            args=self.__build_training_args(),
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=compute_metrics
        )

        best_run = self.__trainer.hyperparameter_search(
            direction="minimize", # Minimize eval_loss
            hp_space=optuna_hp_space,
            n_trials=20
        )
        print(best_run)

    def __build_training_args(self):
        return TrainingArguments(
            num_train_epochs=self.__settings.num_train_epochs,
            learning_rate=self.__settings.learning_rate,
            lr_scheduler_type=self.__settings.lr_scheduler_type,
            warmup_ratio=self.__settings.warmup_ratio,
            metric_for_best_model=self.__settings.metric_for_best_model,
            greater_is_better=self.__settings.greater_is_better,
            weight_decay=self.__settings.weight_decay,
            label_smoothing_factor=self.__settings.label_smoothing_factor,
            gradient_accumulation_steps=self.__settings.gradient_accumulation_steps,
            per_device_train_batch_size=self.__settings.batch_size,
            per_device_eval_batch_size=self.__settings.batch_size,
            seed=self.__settings.random_seed,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            output_dir=self.__settings.ouput_dir
        )

    def __model_init(self):
        config = AutoConfig.from_pretrained(self._model_path)
        config.hidden_dropout_prob = self.__settings.dropout_prob
        config.attention_probs_dropout_prob = self.__settings.dropout_prob

        self._model = BertForSequenceClassification.from_pretrained(self._model_path, config=config)

        # Freeze all base BERT parameters
        for param in self._model.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last n layers
        for i in range(-self.__settings.n_layer_unfreeze, 0):
            for param in self._model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        # Unfreeze the classifier layer
        for param in self._model.classifier.parameters():
            param.requires_grad = True

        print("Trainable parameters:")
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                print(name)

    class FocalLossTrainer(Trainer):
        def __init__(self, alpha=None, gamma=2.0, **kwargs):
            kwargs['compute_metrics'] = self.__compute_metrics
            super().__init__(**kwargs)
            self.alpha = alpha
            self.gamma = gamma
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")

                if self.alpha is not None:
                    alpha_tensor = self.alpha.to(logits.device)
                else:
                    alpha_tensor = None
                    
                loss = self.__focal_loss(
                    logits=logits, 
                    targets=labels, 
                    alpha=alpha_tensor, 
                    gamma=self.gamma
                )

                return (loss, outputs) if return_outputs else loss
        
        """
        FL = -alpha_t * (1 - pt)**gamma * log(pt)
    
        pt: The predicted probability for the true class.
        log(pt): Standard Cross-Entropy loss (CE).
        (1 - pt)**gamma: The modulating factor. gamma controls the strength of down-weighting for easy instances.
        alpha_t: A class-specific weighting factor (alpha) to balance loss across positive/negative classes (handles data imbalance).
    
        Since CE = -log(pt), the formula is often written as:
        FL = alpha_t * (1 - pt)**gamma * CE
        """
        def __focal_loss(self, logits, targets, alpha=None, gamma=2.0):
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            modulating_factor = (1 - pt)**gamma
            loss = modulating_factor * ce_loss
            
            if alpha is not None:
                alpha_t = alpha[targets]
                loss = alpha_t * loss
                
            return loss.mean()
        
        def __compute_metrics(self, p):
            logits, labels = p
            pred = np.argmax(logits, axis=1)
            
            accuracy = accuracy_score(labels, pred)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='weighted')
            
            precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(labels, pred, average=None)

            # Convert logits to probabilities using softmax
            y_pred_proba = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))[:, 1]
            ece = get_ece(y_pred_proba, labels)

            return {
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
                "ece": ece
            }