from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import List, Literal, Optional
from model.bert.bert_classifier import BERTClassifier
from model.protocols import ModelContainer, ModelSettings, TrainTestSplit
from data_loader import DataLoader


class BERTModelContainer(ModelContainer):
    SUPPORTED_CALIBRATION_METHODS = ('sigmoid', 'isotonic', None)
    data: Optional[TrainTestSplit]

    def __init__(self, loader: DataLoader, model: BERTClassifier, settings=ModelSettings(),
                calibration_method: Literal['sigmoid', 'isotonic', None] = 'sigmoid'):
        if calibration_method not in self.SUPPORTED_CALIBRATION_METHODS:
            raise ValueError(f"Invalid calibration_method: '{calibration_method}'.")
        
        self.loader = loader
        self.model = model
        self.settings = settings
        self.calibration_method = calibration_method
        self.calibration_ratio = None if calibration_method is None else 0.5


    def train(self) -> None:
        self.data = self.loader.load_data(self.calibration_ratio)
        dict = self.loader.load_data_dict(self.data)
        self.model.train(dict)
        if self.calibration_method is not None:
            self.__init_calibrator()

    def test(self) -> None:
        if (self.data is None):
            self.data = self.loader.load_data(self.calibration_ratio)

        y_pred_proba = self.predict_list(self.data.X_test)[:, 1]
        y_pred = (y_pred_proba >= self.settings.threshold).astype(int)

        self._output_metrics(y_pred, y_pred_proba)


    def predict_list(self, X: List[str]) -> np.ndarray:
        proba = self.model.predict_proba(X)
        if self.calibration_method is not None:
            proba = proba[:, 1]
            if self.calibration_method == 'sigmoid':
                p_cal_positive = self.calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
            else:
                p_cal_positive = self.calibrator.transform(proba)
            p_cal_negative = 1.0 - p_cal_positive
            proba = np.column_stack((p_cal_negative, p_cal_positive))
        return proba


    def __init_calibrator(self):
        uncalibrated_probs = self.model.predict_proba(self.data.X_cal)[:, 1]

        if self.calibration_method == 'sigmoid':
            self.calibrator = LogisticRegression()
            self.calibrator.fit(uncalibrated_probs.reshape(-1, 1), self.data.y_cal)
            
        elif self.calibration_method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(uncalibrated_probs, self.data.y_cal)