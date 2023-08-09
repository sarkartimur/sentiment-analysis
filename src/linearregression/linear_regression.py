"""
A reference 'from scratch' implementation of linear regression, using gradient descent for minimization of loss.

Reference material:
https://vitalflux.com/mean-square-error-r-squared-which-one-to-use/
https://vitalflux.com/interpreting-f-statistics-in-linear-regression-formula-examples/
"""
from dataclasses import InitVar, dataclass
import logging
import numpy as np
import scipy.stats


logger = logging.getLogger(__name__)


@dataclass
class GradLinearRegression:
    learn_rate: InitVar[float]
    epochs: InitVar[int]

    slope: np.ndarray[int, np.dtype[np.float64]] = np.random.uniform(0, 1, (2))
    intercept: float = np.random.uniform(0, 100)

    def __post_init__(self, learn_rate, epochs):
        self.__learn_rate = learn_rate
        self.__epochs = epochs

    def fit(self, predictor, target):
        for i in range(self.__epochs):
            if (i % 10000 == 0):
                logger.debug(f'Epoch: {i}, current slope: {self.slope}, current intercept: {self.intercept}')
            self.__gradient_descent(predictor, target)
        logger.info(f'The p value is {self.__calculate_p(predictor, target)}')
        logger.info(f'Slope: {self.slope}, Intercept: {self.intercept}')
        return self

    def plot_line(self, predictor):
        return [self.slope*x + self.intercept for x in range(0, predictor[-1])]

    def __gradient_descent(self, predictor, target):
        """Attempts to minimize loss by way of gradient descent (loss function in this case is mean squared error(MSE))."""
        line = (self.slope*predictor).sum(1) + self.intercept
        # calculate partial derivatives of MSE with respect to slope and intercept
        slope_pd = (-2*predictor*(target.reshape(-1, 1) - line.reshape(-1, 1))).sum(0)
        intercept_pd = (-2*(target - line)).sum()

        # adjust parameters by negative gradient times the learning rate
        self.slope -= slope_pd*self.__learn_rate
        self.intercept -= intercept_pd*self.__learn_rate

    def __calculate_p(self, predictor, target):
        mean = target.mean()
        se_mean = ((target - mean)**2).sum()
        line = (self.slope*predictor).sum(1) + self.intercept
        se_line = ((target - line)**2).sum()
        # Degrees of freedom in the denominator,
        # number of observations minus extra parameters in the model (slope and intercept)
        d_dof = len(target) - len(self.slope) - 1
        var_explained = se_mean-se_line
        var_ratio = var_explained / (se_line/d_dof)
        # Note: degrees of freedom in the numerator are equal to number of params in the model
        # (len(slope) + 1 for intercept) minus number of params without a model (just intercept)
        return 1 - scipy.stats.f.cdf(var_ratio, len(self.slope), d_dof)
