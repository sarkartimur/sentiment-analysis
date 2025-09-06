import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline


RANDOM_SEED = 42


data = fetch_openml(name='creditcard', version=1, as_frame=True)
df = data.frame

print(df.head())

X = df.drop('Class', axis=1)
y = df['Class'].astype(int)
print(f"Dataset shape: {X.shape}")
print(f"Fraud cases: {y.sum()} ({y.sum()/len(y)*100:.3f}% of total)")

# use stratification due to extreme imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
)
print(f"Train fraud cases: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.3f}%)")
print(f"Test fraud cases: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.3f}%)")

"""
Build and run CV
"""
pipeline = ImbPipeline([
    ('scaler', RobustScaler()),
    # ('smote', SMOTE(random_state=RANDOM_SEED)),
    ('classifier', LogisticRegression(random_state=RANDOM_SEED))
])
param_grid = [
    {
        'classifier': [DecisionTreeClassifier(random_state=RANDOM_SEED)],
        'classifier__max_depth': [5, 10],
        # 'smote__sampling_strategy': [0.1, 0.2]
    },
    {
        'classifier': [LogisticRegression(random_state=RANDOM_SEED)],
        'classifier__C': [0.01, 1],
        'classifier__penalty': ['l2'],
        # 'smote__sampling_strategy': [0.1, 0.2]
    },
    {
        'classifier': [XGBClassifier(
            objective='binary:logistic',
            missing=np.nan,
            seed=RANDOM_SEED,
            subsample=0.9,
            colsample_bytree=0.5
        )],
        # 'smote__sampling_strategy': [0.1, 0.2]
    },
]
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    verbose=3,
    n_jobs=-1,
    return_train_score=True
)
grid_search.fit(X_train, y_train)
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation ROC AUC: {grid_search.best_score_:.4f}")

"""
Run test set on best model
"""
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
test_f1 = f1_score(y_test, y_pred)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest ROC AUC: {test_roc_auc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Legit', 'Fraud'],
            yticklabels=['Legit', 'Fraud'])
plt.title('Confusion Matrix - Best Model\n(Fraud Detection)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

"""
Print results
"""
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('mean_test_score', ascending=False)
model_performance = {}
for i, params in enumerate(param_grid):
    model_class = params['classifier'][0].__class__.__name__
    model_mask = results_df['param_classifier'].apply(lambda x: x.__class__.__name__ == model_class)
    model_scores = results_df[model_mask]['mean_test_score']

    if len(model_scores) > 0:
        model_performance[model_class] = {
            'best_roc_auc': model_scores.max(),
            'mean_roc_auc': model_scores.mean(),
            'num_configs': len(model_scores)
        }
print(f"\n{'='*60}")
print("MODEL PERFORMANCE COMPARISON (ROC AUC)")
print(f"{'='*60}")
for model_class, stats in model_performance.items():
    print(f"{model_class}:")
    print(f"  Best ROC AUC: {stats['best_roc_auc']:.4f}")
    print(f"  Mean ROC AUC: {stats['mean_roc_auc']:.4f}")
    print(f"  Configurations tested: {stats['num_configs']}")
    print()

"""
Draw results
"""
model_names = list(model_performance.keys())
best_scores = [stats['best_roc_auc'] for stats in model_performance.values()]
mean_scores = [stats['mean_roc_auc'] for stats in model_performance.values()]
x_pos = np.arange(len(model_names))
plt.bar(x_pos - 0.2, best_scores, 0.4, label='Best ROC AUC', alpha=0.8, color='skyblue')
plt.bar(x_pos + 0.2, mean_scores, 0.4, label='Mean ROC AUC', alpha=0.8, color='lightcoral')
plt.xlabel('Model Type')
plt.ylabel('ROC AUC Score')
plt.xticks(x_pos, model_names, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.8, 1.0)
for i, (best, mean) in enumerate(zip(best_scores, mean_scores)):
    plt.text(i - 0.2, best + 0.005, f'{best:.3f}', ha='center', fontsize=9)
    plt.text(i + 0.2, mean + 0.005, f'{mean:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()
