import numpy as np
from constants import RANDOM_SEED
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from lime.lime_text import LimeTextExplainer
import pickle


def xgboost_cv():
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
        eval_metric='aucpr',
        # majority/minority
        scale_pos_weight=10,
        random_state=RANDOM_SEED
    )
    
    random_search = RandomizedSearchCV(
        xgb_model, param_grid, n_iter=25, scoring='average_precision',
        cv=5, verbose=2, random_state=RANDOM_SEED, n_jobs=-1
    )
    
    return random_search

def svc_cv():
    # best params: C = 100, gamma = 0.01
    param_grid = [
    {
        'C': [0.5, 1, 10, 30, 45, 60, 90, 100],
        'gamma': ['scale', 1, 0.1, 0.01, 0.005, 0.001, 0.0005],
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
    
    return grid_search

def svc():
    model = SVC(random_state=RANDOM_SEED, probability=True, C=60, gamma=0.001)
    model = CalibratedClassifierCV(
        model,
        method='sigmoid', 
        cv=5,
        n_jobs=-1
    )
    return model

def logistic_regression():
    model = CalibratedClassifierCV(
        LogisticRegression(random_state=RANDOM_SEED),
        method='sigmoid', 
        cv=5,
        n_jobs=-1
    )
    return model

def choose_model():
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
    
    return grid_search

def lime_explain(txt, predict_method):
    explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
    exp = explainer.explain_instance(
        txt, 
        predict_method,
        num_features=20,
        num_samples=3000
    )
    # exp.as_pyplot_figure()
    # plt.show()
    exp.show_in_notebook()

def serialize(filename, model):
    with open(filename, 'wb') as f:
        pickle.dump(model, f, protocol=5)

def deserialize(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)