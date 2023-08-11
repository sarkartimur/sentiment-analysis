"""
A reference 'from scratch' implementation of linear regression, using gradient descent for minimization of loss.
Note: gradient descent is extremely inefficient for minimizing error of a linear model, it is used here only for demonstration purpose.

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

    def __post_init__(self, learn_rate, epochs):
        self.__learn_rate = learn_rate
        self.__epochs = epochs

    def fit(self, predictor, target):
        """Attempts to minimize loss by way of batch gradient descent (loss function in this case is square error)."""
        for i in range(self.__epochs):
            logger.debug(f'Epoch: {i}, current slope: {self.slope}')
            self.__gradient_descent(predictor, target)
        logger.info(f'The p value is {self.__calculate_p(predictor, target)}')
        logger.info(f'Slope: {self.slope}')
        return self

    def plot_line(self, predictor):
        return [self.slope*x for x in range(0, predictor[-1])]

    def __gradient_descent(self, predictor, target):
        line = (self.slope*predictor).sum(1)
        # calculate partial derivatives of MSE with respect to each parameter (slope)
        slope_pd = (-2*predictor*(target.reshape(-1, 1) - line.reshape(-1, 1))).sum(0)

        # adjust parameters by negative gradient times the learning rate
        self.slope -= slope_pd*self.__learn_rate

    def __calculate_p(self, predictor, target):
        mean = target.mean()
        sst = ((target - mean)**2).sum()
        plane = (self.slope*predictor).sum(1)
        sse = ((target - plane)**2).sum()
        ssr = sst-sse
        # Degrees of freedom in the numerator,
        # number of coefficients in the model - 1
        ssr_dof = len(self.slope) - 1
        # Degrees of freedom in the denominator,
        # number of observations minus extra parameters in the model (coefficients)
        sse_dof = len(target) - len(self.slope)
        f = (ssr/ssr_dof) / (sse/sse_dof)
        return 1 - scipy.stats.f.cdf(f, ssr_dof, sse_dof)
