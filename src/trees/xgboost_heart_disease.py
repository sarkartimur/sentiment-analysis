from tabnanny import verbose
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo

TARGET_COL = 'hd'
RANDOM_SEED = 42


heart_disease = fetch_ucirepo(id=45)
df = heart_disease.data.original
df.rename(columns={'num': TARGET_COL}, inplace=True)

print(df.head())

df = df.loc[~df['ca'].isna() & ~df['thal'].isna()]

X = df.drop(TARGET_COL, axis=1).copy()
y = df[TARGET_COL].copy()
# binary classification
y[y > 0] = 1

X = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED)

xgbc = xgb.XGBClassifier(
    objective='binary:logistic',
    missing=np.nan,
    seed=RANDOM_SEED,
    early_stopping_rounds=10,
    eval_metric='aucpr'
)
xgbc.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

ConfusionMatrixDisplay.from_estimator(xgbc, X_test, y_test, normalize='all')


param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05, 0.5, 1],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10, 20, 100],
    'scale_pos_weight': [1, 3, 5]
}
grid_search = GridSearchCV(
    xgb.XGBClassifier(
        objective='binary:logistic',
        missing=np.nan,
        seed=RANDOM_SEED,
        subsample=0.9,
        colsample_bytree=0.5
    ),
    param_grid,
    scoring='roc_auc',
    n_jobs=16,
    verbose=3
)
grid_search.fit(X_train, y_train)

print(f'Best params: {grid_search.best_params_}')
ConfusionMatrixDisplay.from_estimator(grid_search.best_estimator_, X_test, y_test, normalize='all')