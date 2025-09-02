import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

dtc = DecisionTreeClassifier(random_state=RANDOM_SEED)
dtc.fit(X_train, y_train)

plt.figure(figsize=(50, 25))
plot_tree(dtc, filled=True, rounded=True, class_names=['Healthy', 'Not healthy'], feature_names=X.columns)
ConfusionMatrixDisplay.from_estimator(dtc, X_test, y_test, normalize='all')

"""
Optimization (cost-complexity pruning)
"""
path = dtc.cost_complexity_pruning_path(X_train, y_train)

plt.figure()
plt.plot(path.ccp_alphas, path.impurities, marker='o', drawstyle="steps-post")
plt.xlabel("effective alpha")
plt.ylabel("total impurity of leaves") # for binary classification 0.5 - random guessing
plt.show()

# remove maximum alpha (it corresponds to a single-node tree with impurity = 0.5)
ccp_alphas = path.ccp_alphas[:-1]

dtcs = []
for a in ccp_alphas:
    dtc = DecisionTreeClassifier(random_state=RANDOM_SEED, ccp_alpha=a)
    dtc.fit(X_train, y_train)
    dtcs.append(dtc)

train_scores = [dt.score(X_train, y_train) for dt in dtcs]
test_scores = [dt.score(X_test, y_test) for dt in dtcs]

plt.figure()
plt.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle="steps-post")
plt.xlabel("effective alpha")
plt.ylabel("score")
plt.legend()
plt.show()

"""
Optimization (cross-validation)
"""
param_grid = {'ccp_alpha': ccp_alphas}
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_SEED),
    param_grid,
    cv=10,
    verbose=3
)
grid_search.fit(X_train, y_train)

print(f"Optimal ccp_alpha: {grid_search.best_params_['ccp_alpha']}")
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test set accuracy of the best model: {test_score:.4f}")
plt.figure(figsize=(50, 25))
plot_tree(best_model, filled=True, rounded=True, class_names=['Healthy', 'Not healthy'], feature_names=X.columns)
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, normalize='all')