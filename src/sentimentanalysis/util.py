import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.decomposition import PCA
from umap import UMAP


RANDOM_SEED = 42


def load_data(sample_size=2000, test_ratio=0.2):
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')
    df = pd.concat([dataset['train'].to_pandas(), dataset['test'].to_pandas()], ignore_index=True)
    
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=sample_size,
        test_size=int(sample_size * test_ratio),
        stratify=y,
        random_state=RANDOM_SEED
    )
    
    print(f"Training set Positive: {sum(y_train)}, Negative: {len(y_train) - sum(y_train)}")
    print(f"Testing set Positive: {sum(y_test)}, Negative: {len(y_test) - sum(y_test)}")
    
    return X_train, y_train, X_test, y_test

def train_xgboost(X_train, y_train, X_val, y_val):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED
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
    
def evaluate_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred, y_pred_proba

def reduce_dimensions(data, n_components=50, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'umap':
        reducer = UMAP(n_components=n_components, random_state=RANDOM_SEED)
    else:
        raise ValueError("Method must be 'pca' or 'umap'")
    
    print(f"Reducing dimensions using {method}")
    reduced = reducer.fit_transform(data)
    return reduced, reducer