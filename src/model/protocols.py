from typing import Protocol, Union, Optional, Iterator, Tuple, List
from dataclasses import dataclass, astuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from constants import RANDOM_SEED
import metrics


Features = Union[np.ndarray, pd.DataFrame, pd.Series]
Target = Union[np.ndarray, pd.Series]


@dataclass(frozen=True)
class TrainTestSplit:
    X_train: Features
    y_train: Target
    X_test: Optional[Features] = None
    y_test: Optional[Target] = None
    X_cal: Optional[Features] = None
    y_cal: Optional[Target] = None

    def __iter__(self) -> Iterator:
        return iter(astuple(self))
    
    @staticmethod
    def _concat(a: Optional[Features], b: Optional[Features]) -> Optional[Features]:
        if a is None and b is None:
            return None
        return pd.concat([a, b], ignore_index=True)
    
    @staticmethod
    def _shuffle(X: Optional[Features], y: Optional[Target]) -> Optional[Tuple[Features, Target]]:
        if X is None and y is None:
            return [None, None]
        return shuffle(X, y, random_state=RANDOM_SEED)

    @classmethod
    def merge(cls: type['TrainTestSplit'], split1: 'TrainTestSplit', split2: 'TrainTestSplit') -> 'TrainTestSplit':
        X_train_merged = cls._concat(split1.X_train, split2.X_train)
        y_train_merged = cls._concat(split1.y_train, split2.y_train)

        X_test_merged = cls._concat(split1.X_test, split2.X_test)
        y_test_merged = cls._concat(split1.y_test, split2.y_test)

        X_cal_merged = cls._concat(split1.X_cal, split2.X_cal)
        y_cal_merged = cls._concat(split1.y_cal, split2.y_cal)

        X_train_merged, y_train_merged = cls._shuffle(X_train_merged, y_train_merged)
        X_test_merged, y_test_merged = cls._shuffle(X_test_merged, y_test_merged)
        X_cal_merged, y_cal_merged = cls._shuffle(X_cal_merged, y_cal_merged)
        
        return cls(
            X_train_merged, y_train_merged,
            X_test_merged, y_test_merged,
            X_cal_merged, y_cal_merged
        )


@dataclass(frozen=True)
class ModelSettings:
    do_reduce = True
    do_output_metrics = True
    do_analyze_errors = True
    bert_pooling = 'mean'
    num_dimensions = 150
    threshold = 0.5


class Classifier(Protocol):
    def fit(self, X_train: Features, y_train: Target) -> 'Classifier':
        return self

    def predict(self, X: Features) -> Target:
        pass

    def predict_proba(self, X: Features) -> np.ndarray:
        pass


# todo add method fromfile
class ModelContainer(ABC):
    settings: ModelSettings
    model: Classifier
    data: Optional[TrainTestSplit] = None
    cfp: Optional[Features] = None
    cfn: Optional[Features] = None

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def test(self) -> None:
        pass

    @abstractmethod
    def predict_list(self, X: List[str]) -> np.ndarray:
        pass

    def predict(self, text: str) -> np.ndarray:
        return self.predict_list([text])
    
    def _analyze_errors(self, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> None:
        if self.data is not None:
            self.cfp = self.data.X_test[(self.data.y_test == 0) & (y_pred == 1) & (y_pred_proba > 0.9)]
            self.cfn = self.data.X_test[(self.data.y_test == 1) & (y_pred == 0) & (y_pred_proba < 0.1)]
            
            print("\n")
            for i, fn in self.cfp[:10].items():
                print(f"Confident false positive at index {i}: {fn}")

            print("\n")
            for i, fn in self.cfn[:10].items():
                print(f"Confident false negative at index {i}: {fn}")

    def _output_metrics(self, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> None:
        if self.settings.do_output_metrics:
            metrics.compute_metrics(self.data.y_test, y_pred, y_pred_proba)
            metrics.plot_graphs(self.data.y_test, y_pred, y_pred_proba)

        if self.settings.do_analyze_errors:
            self._analyze_errors(y_pred, y_pred_proba)