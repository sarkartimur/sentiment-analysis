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

def load_imdb_data(sample_size=2000):
    """Load and sample IMDB dataset"""
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')
    
    # Sample data for faster processing (remove for full dataset)
    train_texts = dataset['train']['text'][:sample_size]
    train_labels = dataset['train']['label'][:sample_size]
    test_texts = dataset['test']['text'][:sample_size//2]
    test_labels = dataset['test']['label'][:sample_size//2]
    
    return train_texts, train_labels, test_texts, test_labels

def visualize_embeddings(embeddings, labels, title="Embeddings Visualization"):
    """Visualize reduced embeddings using UMAP"""
    reducer = UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()

def main():
    # Configuration
    SAMPLE_SIZE = 2000  # Reduce for faster testing, increase for better results
    EMBEDDING_DIM = 50
    RANDOM_STATE = 42
    
    print("Starting BERT + XGBoost Sentiment Analysis...")
    
    # Load data
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(SAMPLE_SIZE)
    
    # Initialize BERT classifier
    classifier = BERTSentimentClassifier()
    
    # Extract BERT embeddings
    print("\nExtracting BERT embeddings for training data...")
    start_time = time.time()
    train_embeddings = classifier.get_bert_embeddings(train_texts)
    print(f"Embedding extraction time: {time.time() - start_time:.2f} seconds")
    
    print("\nExtracting BERT embeddings for test data...")
    test_embeddings = classifier.get_bert_embeddings(test_texts)
    
    # Reduce dimensions
    print(f"\nReducing dimensions to {EMBEDDING_DIM} using PCA...")
    train_embeddings_reduced, pca_reducer = classifier.reduce_dimensions(
        train_embeddings, n_components=EMBEDDING_DIM, method='pca'
    )
    test_embeddings_reduced = pca_reducer.transform(test_embeddings)
    
    # Visualize reduced embeddings
    visualize_embeddings(train_embeddings_reduced, train_labels, "Training Embeddings (PCA Reduced)")
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_embeddings_reduced, train_labels, 
        test_size=0.2, random_state=RANDOM_STATE, stratify=train_labels
    )
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    xgb_model = classifier.train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred, y_pred_proba = classifier.evaluate_model(xgb_model, test_embeddings_reduced, test_labels)
    
    # Feature importance
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(xgb_model, max_num_features=20)
    plt.title('Top 20 Feature Importances')
    plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return xgb_model, pca_reducer, classifier

if __name__ == "__main__":
    model, pca, classifier = main()