"""
A reference 'from scratch' implementation of linear regression, using gradient descent for minimization of loss.

Reference material:
https://vitalflux.com/mean-square-error-r-squared-which-one-to-use/
https://vitalflux.com/interpreting-f-statistics-in-linear-regression-formula-examples/
"""
from dataclasses import InitVar, dataclass
import logging
import numpy as np
from . import util as u


logger = logging.getLogger(__name__)


@dataclass
class GradLinearRegression:
    learn_rate: InitVar[float]
    epochs: InitVar[int]

    slope: np.ndarray[int, np.dtype[np.float64]] = np.empty(0)
    intercept: float = 0

    def __post_init__(self, learn_rate, epochs):
        self.__learn_rate = learn_rate
        self.__epochs = epochs

    def fit(self, predictor, target):
        for i in range(self.__epochs):
            if (i % 10000 == 0):
                logger.debug(f'Epoch: {i}, current slope: {self.slope}, current intercept: {self.intercept}')
            self.__gradient_descent(predictor, target)
        # logger.info(f'The p value is {self.__calculate_p(predictor, target)}')
        logger.info(f'Slope: {self.slope}, Intercept: {self.intercept}')
        return self

    def plot_line(self, predictor):
        return [self.slope*x + self.intercept for x in range(0, predictor[-1])]

    def __gradient_descent(self, predictor: np.ndarray[int, np.dtype[np.float64]],
                           target: np.ndarray[int, np.dtype[np.float64]]) -> None:
        """Attempts to minimize loss by way of gradient descent (loss function in this case is mean squared error(MSE)).

        Returns
        -------
            Parameters (slope and intercept) adjusted by negative gradient times the learning rate
        """
        if (not self.slope.any()):
            self.slope = np.zeros(predictor.shape[1])
        
        line = (self.slope*predictor).sum(1) + self.intercept
        # calculate partial derivatives of MSE with respect to slope and intercept
        slope_pd = (-2*predictor*(target.reshape(-1, 1) - line.reshape(-1, 1))).sum(0)
        intercept_pd = (-2*(target - line)).sum()

        self.slope -= slope_pd*self.__learn_rate
        self.intercept -= intercept_pd*self.__learn_rate

    def __calculate_p(self, predictor, target):
        se_mean = u.squared_error_total(target)
        se_regression = u.squared_error_regression(predictor, target, self.slope, self.intercept)
        # Degrees of freedom in the denominator,
        # number of observations minus 2 extra parameters in the model (slope and intercept)
        d_dof = len(target) - 2
        var_explained = se_mean-se_regression
        var_ratio = var_explained / (se_regression/d_dof)
        # Note: degrees of freedom in the numerator are equal to 1 in the case of 2d linear model
        return u.f_test(var_ratio, 1, d_dof)
