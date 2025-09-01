import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE
from collections import Counter

# Note: there are 50 samples of each of 3 classes, so 50 max
SAMPLE_COUNT = 50
REMOVE_COUNT = 40
REMOVE_CLASS = 'Iris-setosa'
RANDOM_SEED = 42


# https://archive.ics.uci.edu/dataset/53/iris
iris = fetch_ucirepo(id=53)
df: pd.DataFrame = iris.data.original
df = df.groupby('class').sample(n=SAMPLE_COUNT, random_state=RANDOM_SEED)


"""
Remove some samples
"""
to_remove = df[df['class'] == REMOVE_CLASS]
to_remove = to_remove.sample(n=REMOVE_COUNT, random_state=RANDOM_SEED)
df = df.drop(to_remove.index)

"""
Split data into predictors and target
"""
X = df.drop('class', axis=1).copy()
y = df['class'].copy()

"""
Oversample using smote
"""
print(f"Original class distribution: {Counter(y)}")
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
print(f"Resampled class distribution: {Counter(y)}")

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