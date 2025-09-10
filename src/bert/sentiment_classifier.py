import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import time

class BERTSentimentClassifier:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
    def get_bert_embeddings(self, texts, batch_size=16, max_length=512):
        """Extract BERT embeddings for a list of texts"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Mean pooling (average of all token embeddings)
                # batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                # Max pooling (take maximum values across tokens)
                # batch_embeddings = outputs.last_hidden_state.max(dim=1).values.cpu().numpy()
                # Use [CLS] token embedding as sentence representation
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i}/{len(texts)} texts")
        
        return np.vstack(embeddings)
    
    def reduce_dimensions(self, embeddings, n_components=50, method='umap'):
        """Reduce embedding dimensions"""
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'umap':
            reducer = UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError("Method must be 'pca' or 'umap'")
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings, reducer
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost classifier"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        eval_list = [(dtrain, 'train'), (dval, 'eval')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=eval_list,
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the trained model"""
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return y_pred, y_pred_proba

def load_imdb_data(sample_size=2000, test_ratio=0.2, random_state=42):
    """Load IMDB dataset using Hugging Face's stratification"""
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')
    
    # Create stratified split for training data
    train_split = dataset['train'].train_test_split(
        test_size=test_ratio, 
        stratify_by_column='label',
        seed=random_state
    )
    
    # Create stratified split for test data
    test_split = dataset['test'].train_test_split(
        test_size=test_ratio,  # Half of test data for validation
        stratify_by_column='label',
        seed=random_state
    )
    
    # Get the data
    train_texts = train_split['train']['text'][:sample_size]
    train_labels = train_split['train']['label'][:sample_size]
    test_texts = test_split['test']['text'][:sample_size//2]
    test_labels = test_split['test']['label'][:sample_size//2]
    
    # Verify the distribution
    print(f"Training set - Positive: {sum(train_labels)}, Negative: {len(train_labels) - sum(train_labels)}")
    print(f"Test set - Positive: {sum(test_labels)}, Negative: {len(test_labels) - sum(test_labels)}")
    
    return train_texts, train_labels, test_texts, test_labels