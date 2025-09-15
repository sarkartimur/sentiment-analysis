import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from datasets import DatasetDict


RANDOM_SEED = 42


def load_data(sample_size=2000, test_ratio=0.25, minority_class=0, imbalance_ratio=1.0):
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')
    df = pd.concat([dataset['train'].to_pandas(), dataset['test'].to_pandas()], ignore_index=True)

    print(df.head())
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=sample_size,
        test_size=int(sample_size * test_ratio),
        stratify=y,
        random_state=RANDOM_SEED
    )
    
    print(f"Original training set - Positive: {sum(y_train)}, Negative: {len(y_train) - sum(y_train)}")
    
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    
    majority_df = train_df[train_df['label'] != minority_class]
    minority_df = train_df[train_df['label'] == minority_class]
    
    target_minority_count = int(len(majority_df) * imbalance_ratio)
    if len(minority_df) > target_minority_count:
        minority_df = minority_df.sample(n=target_minority_count, random_state=RANDOM_SEED)
    else:
        print(f"Warning: Cannot achieve {imbalance_ratio} ratio - not enough minority samples")
    
    imbalanced_train_df = pd.concat([majority_df, minority_df])
    # Shuffle
    imbalanced_train_df = imbalanced_train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    X_train_imbalanced = imbalanced_train_df['text']
    y_train_imbalanced = imbalanced_train_df['label']
    
    print(f"Imbalanced training set - Class {minority_class} as minority")
    print(f"Positive: {sum(y_train_imbalanced)}, Negative: {len(y_train_imbalanced) - sum(y_train_imbalanced)}")
    print(f"Imbalance ratio: ~1:{round(1/imbalance_ratio)}")
    print(f"Testing set - Positive: {sum(y_test)}, Negative: {len(y_test) - sum(y_test)}")
    
    return X_train_imbalanced, y_train_imbalanced, X_test, y_test

def load_data_dict(sample_size=2000, test_ratio=0.25):
    full_dataset = load_dataset('imdb')
    small_train = full_dataset['train'].train_test_split(
        train_size=sample_size, 
        test_size=int(sample_size * test_ratio), 
        seed=RANDOM_SEED, 
        stratify_by_column='label'
    )

    return DatasetDict({
        'train': small_train['train'],
        'test': small_train['test']
    })

def train_xgboost(X_train, y_train):
    param_grid = {
        'max_depth': [2, 3, 6, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.3, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.3, 0.6, 0.8, 1.0],
        'n_estimators': [100, 200, 300, 400, 500],
        'reg_alpha': [0, 0.1, 0.5, 0.7],
        'reg_lambda': [1, 1.5, 2, 3, 5]
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_SEED
    )
    
    random_search = RandomizedSearchCV(
        xgb_model, param_grid, n_iter=25, scoring='accuracy',
        cv=5, verbose=2, random_state=RANDOM_SEED, n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print("Best parameters:", random_search.best_params_)
    print("Best cross-validation score:", random_search.best_score_)
    
    return random_search.best_estimator_

def svc_cv(X_train, y_train):
    # best params: C = 100, gamma = 0.01
    param_grid = [
    {
        'C': [0.5, 1, 10, 100],
        'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }]
    grid_search = GridSearchCV(
        SVC(random_state=RANDOM_SEED, probability=True),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='roc_auc',
        verbose=3
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def train_svc(X_train, y_train):
    model = CalibratedClassifierCV(
        SVC(random_state=RANDOM_SEED, probability=True, C=10, gamma=0.01),
        method='sigmoid',
        cv=5,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = CalibratedClassifierCV(
        LogisticRegression(random_state=RANDOM_SEED),
        method='sigmoid', 
        cv=5,
        n_jobs=-1
    )
    # model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model

def choose_model(X_train, y_train):
    pipeline = ImbPipeline([
        ('classifier', LogisticRegression(random_state=RANDOM_SEED))
    ])
    param_grid = [
        {
            'classifier': [DecisionTreeClassifier(random_state=RANDOM_SEED)],
            'classifier__max_depth': [3, 5, 10]
        },
        {
            'classifier': [LogisticRegression(random_state=RANDOM_SEED)],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'classifier__penalty': ['l2']
        },
        {
            'classifier': [SVC(random_state=RANDOM_SEED)],
            'classifier__C': [0.5, 1, 10, 100],
            'classifier__gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
            'classifier__kernel': ['rbf']
        },
    ]
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        verbose=3,
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

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

def analyze_errors(y_test, y_pred, test_texts):
    """Analyze where the model is making errors"""
    incorrect_indices = np.where(y_pred != y_test)[0]
    
    # Look at some misclassified examples
    for i in incorrect_indices[:10]:
        print(f"\nIndex: {i}, True label: {y_test[i]}, Predicted: {y_pred[i]}")
        print(f"Text: {test_texts[i][:200]}...")
    
    return incorrect_indices