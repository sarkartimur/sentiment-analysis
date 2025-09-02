import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo


SAMPLE_COUNT = 1000
RANDOM_SEED = 42


"""
Load data
Default of Credit Card Clients dataset
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
"""
dataset = fetch_ucirepo(id=350) 

print(dataset.metadata)
print(dataset.variables)

df: pd.DataFrame = dataset.data.original

"""
Cleaning
For some reason features X3 (education) and X4 (marital status)
contain value 0 which is not in the specification.
Remove these samples.
"""
df = df.loc[(df.X3 != 0) & (df.X4 != 0)] # 29932 samples

"""
Downsampling
"""
df_no_default = df[df.Y == 0] # 6631
df_default = df[df.Y == 1] # 23301

# Note: set replace=True to reuse samples
df_no_default = pd.DataFrame(resample(df_no_default, replace=False, n_samples=SAMPLE_COUNT, random_state=RANDOM_SEED))
df_default = pd.DataFrame(resample(df_default, replace=False, n_samples=SAMPLE_COUNT, random_state=RANDOM_SEED))

df = pd.concat(objs=[df_no_default, df_default])

"""
Split data into predictors and target
"""
X = df.drop('Y', axis=1).copy()
y = df.Y.copy()

"""
Transformation
Apply one-hot encoding to categorical features
(sex, education, marital status, pay 0 through 6),
split data into train/test sets,
standardize.
"""
X = pd.get_dummies(X, columns=['X2', 'X3', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED)
X_train = scale(X_train)
X_test = scale(X_test)

"""
Build and train a support vector classifier
"""
svc = SVC(random_state=RANDOM_SEED)
svc.fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(svc, X_test, y_test, normalize='all')

"""
Optimize model using cross validation
"""
param_grid = [
    {
        'C': [0.5, 1, 10, 100], # default = 1
        'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001], # default = 'scale'
        'kernel': ['rbf']
    }
]
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    verbose=3
)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test set accuracy of the best model: {test_score:.4f}")
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, normalize='all')