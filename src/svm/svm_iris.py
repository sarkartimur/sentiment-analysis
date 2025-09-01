import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo

# Note: there are 50 samples of each of 3 classes, so 50 max
SAMPLE_COUNT = 50
RANDOM_SEED = 42


# https://archive.ics.uci.edu/dataset/53/iris
iris = fetch_ucirepo(id=53)
df: pd.DataFrame = iris.data.original
# df = df.groupby('class').sample(n=SAMPLE_COUNT, random_state=RANDOM_SEED)

"""
Split data into predictors and target
"""
X = df.drop('class', axis=1).copy()
y = df['class'].copy()

"""
Split data into train/test sets, standardize.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED)
X_train = scale(X_train)
X_test = scale(X_test)

"""
Build and train a support vector classifier
"""
svc = SVC(random_state=RANDOM_SEED)
svc.fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(svc, X_test, y_test, normalize='all')