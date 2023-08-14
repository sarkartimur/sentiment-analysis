"""
A reference 'from scratch' implementation of linear regression, using gradient descent for minimization of loss.
Note: gradient descent is extremely inefficient for minimizing error of a linear model, it is used here only for demonstration purpose.

Reference material:
https://vitalflux.com/interpreting-f-statistics-in-linear-regression-formula-examples/
"""
from dataclasses import InitVar, dataclass
import logging
import numpy as np
import scipy.stats
from . import gradient_descent as gd


logger = logging.getLogger(__name__)


@dataclass
class GradLinearRegression:
    epochs: InitVar[int]
    gd_strategy: InitVar[gd.IGradientDescentStrategy]

    p_value: float = 0
    slope: np.ndarray[int, np.dtype[np.float64]] = np.random.uniform(0, 1, (2))

    def __post_init__(self, epochs, gd_strategy):
        self.__epochs = epochs
        self.__gd_strategy = gd_strategy

    def fit(self, predictor, target):
        """Attempts to minimize loss by way of batch gradient descent."""
        for i in range(self.__epochs):
            logger.debug(f'Epoch: {i}, current slope: {self.slope}')
            self.__gd_strategy.gradient_descent(self, predictor, target)

        self.__calculate_p(predictor, target)
        logger.info(f'Slope: {self.slope}, p-value: {self.p_value}')
        return self

    def plot_line(self, predictor):
        return [self.slope*x for x in range(0, predictor[-1])]

    def __calculate_p(self, predictor, target):
        mean = target.mean()
        sst = ((target - mean)**2).sum()
        plane = (self.slope*predictor).sum(1)
        sse = ((target - plane)**2).sum()
        ssr = sst-sse
        # Degrees of freedom in the numerator,
        # number of coefficients in the model
        ssr_dof = len(self.slope)
        # Degrees of freedom in the denominator,
        # number of observations minus extra parameters in the model (coefficients).
        # Note: 1 is not subtracted because intercept is not approximated.
        sse_dof = len(target) - len(self.slope)
        f = (ssr/ssr_dof) / (sse/sse_dof)
        self.p_value = float(1 - scipy.stats.f.cdf(f, ssr_dof, sse_dof))
