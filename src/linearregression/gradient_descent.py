"""
Gradient descent strategy interface and implementations.

Reference material:
https://vitalflux.com/mean-square-error-r-squared-which-one-to-use/
"""
import abc
import numpy as np


class IGradientDescentStrategy(metaclass=abc.ABCMeta):
    _learn_rate: float

    def __init__(self, learn_rate):
        self._learn_rate = learn_rate

    @abc.abstractmethod
    def gradient_descent(self, model, predictor: np.ndarray, target: np.ndarray):
        raise NotImplementedError


class SquareErrorGradientDescentStrategy(IGradientDescentStrategy):
    def gradient_descent(self, model, predictor: np.ndarray, target: np.ndarray):
        """Square error loss function"""
        line = (model.slope*predictor).sum(1)
        # calculate partial derivatives of MSE with respect to each parameter (slope)
        slope_pd = (-2*predictor*(target.reshape(-1, 1) - line.reshape(-1, 1))).sum(0)

        # adjust parameters by negative gradient times the learning rate
        model.slope -= slope_pd*self._learn_rate
