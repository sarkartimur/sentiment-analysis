"""
Contains a simple reference 'from scratch' implementation of single-variate linear regression.

Reference material:
https://vitalflux.com/mean-square-error-r-squared-which-one-to-use/
https://vitalflux.com/interpreting-f-statistics-in-linear-regression-formula-examples/
"""
from dataclasses import InitVar, dataclass
import logging
import numpy as np
import scipy
from .util import *


logger = logging.getLogger(__name__)


@dataclass
class SimpleLinearRegression:
    predictor: InitVar[np.ndarray]
    target: InitVar[np.ndarray]
    learn_rate: InitVar[float]
    epochs: InitVar[int]

    slope: float = 0
    intercept: float = 0

    def __post_init__(self, predictor, target, learn_rate, epochs):
        self.__predictor = predictor
        self.__target = target
        self.__learn_rate = learn_rate
        self.__epochs = epochs

    def fit(self):
        for i in range(self.__epochs):
            if (i % 1000 == 0):
                logger.debug(f'Epoch: {i}, current slope: {self.slope}, current intercept: {self.intercept}')
            self.__gradient_descent()
        logger.info(f'The p value is {self.__calculate_p()}')
        logger.info(f'Slope: {self.slope}, Intercept: {self.intercept}')
        return self

    def plot_line(self):
        return [self.slope*x + self.intercept for x in range(0, self.__predictor[-1])]

    def __gradient_descent(self):
        """Attempts to minimize loss by way of gradient descent (loss function in this case is mean squared error(MSE)).

        Returns
        -------
            Parameters (slope and intercept) adjusted by negative gradient times the learning rate
        """
        slope_pd = 0
        intercept_pd = 0
        for x, y in zip(self.__predictor, self.__target):
            line = self.slope*x + self.intercept
            # calculate partial derivatives of MSE with respect to slope and intercept
            slope_pd += -2*x*(y-line)
            intercept_pd += -2*(y-line)

        self.slope -= slope_pd*self.__learn_rate
        self.intercept -= intercept_pd*self.__learn_rate

    def __calculate_p(self):
        se_mean = squared_error_total(self.__target)
        se_regression = squared_error_regression(self.__predictor, self.__target, self.slope, self.intercept)
        # Degrees of freedom in the denominator,
        # number of observations minus 2 extra parameters in the model (slope and intercept)
        d_dof = len(self.__target) - 2
        var_explained = se_mean-se_regression
        var_ratio = var_explained / (se_regression/d_dof)
        # Note: degrees of freedom in the numerator are equal to 1 in the case of 2d linear model
        return f_test(var_ratio, 1, d_dof)
