from typing import Protocol, Union, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd


Features = Union[np.ndarray, pd.DataFrame, pd.Series]
Target = Union[np.ndarray, pd.Series]

class Classifier(Protocol):
    def fit(self, X_train: Features, y_train: Target) -> 'Classifier':
        pass

    def predict(self, X: Features) -> Target:
        pass

    def predict_proba(self, X: Features) -> np.ndarray:
        pass

@dataclass(frozen=True)
class TrainTestSplit:
    X_train: Features
    y_train: Target
    X_test: Features
    y_test: Target
    X_cal: Optional[Features] = None
    y_cal: Optional[Target] = None