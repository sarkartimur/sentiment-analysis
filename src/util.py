import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from lime.lime_text import LimeTextExplainer
import pickle


RANDOM_SEED = 42


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
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def train_svc(X_train, y_train, X_cal, y_cal):
    model = SVC(random_state=RANDOM_SEED, probability=True, C=60, gamma=0.001)
    
    # # Stage 1: Sigmoid calibration
    # model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    # model.fit(X_train, y_train)

    # # Stage 2: Isotonic on top of sigmoid
    # model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    
    # iso = IsotonicRegression(out_of_bounds='clip')

    # More aggressive isotonic - allows more extreme adjustments
    iso = IsotonicRegression(out_of_bounds='clip')
                                    # increasing=True,  # Force monotonic
                                    # y_min=0.0, y_max=1.0)  # Full range

    model.fit(X_train, y_train)

    X_cal = model.predict_proba(X_cal)[:, 1]
    iso.fit(X_cal, y_cal)
    
    return model, iso

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

def analyze_errors(y_test, y_pred, test_texts):
    """Analyze where the model is making errors"""
    incorrect_indices = np.where(y_pred != y_test)[0]
    
    # Look at some misclassified examples
    for i in incorrect_indices[:10]:
        print(f"\nIndex: {i}, True label: {y_test[i]}, Predicted: {y_pred[i]}")
        print(f"Text: {test_texts[i][:200]}...")
    
    return incorrect_indices

def serialize(filename, model):
    with open(filename, 'wb') as f:
        pickle.dump(model, f, protocol=5)

def deserialize(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)