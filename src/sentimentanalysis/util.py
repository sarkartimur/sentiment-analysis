import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.model_selection import RandomizedSearchCV


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
    param_grid = {
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators': [100, 200, 300],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        early_stopping_rounds=10
    )
    
    # Use randomized search for faster tuning
    random_search = RandomizedSearchCV(
        xgb_model, param_grid, n_iter=50, scoring='accuracy',
        cv=3, verbose=2, random_state=RANDOM_SEED, n_jobs=-1
    )
    
    random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    print("Best parameters:", random_search.best_params_)
    print("Best cross-validation score:", random_search.best_score_)
    
    return random_search.best_estimator_

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