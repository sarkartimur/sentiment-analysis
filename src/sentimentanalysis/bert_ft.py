import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import util

dataset = util.load_data_dict()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", 
                            columns=["input_ids", "attention_mask", "labels"])


def model_init():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=2
    )
    
    # Freeze all base BERT parameters
    for param in model.bert.parameters():
        param.requires_grad = False

    # Unfreeze the last n layers
    n = 2
    for i in range(-n, 0):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = True

    # Unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
        
    return model

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
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

trainer = Trainer(
    model=model_init(),
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print(f"Final evaluation results: {results}")

trainer.save_model("./imdb_sentiment_reduced")
tokenizer.save_pretrained("./imdb_sentiment_reduced")