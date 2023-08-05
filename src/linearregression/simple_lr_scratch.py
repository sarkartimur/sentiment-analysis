"""
Contains a simple reference 'from scratch' implementation of single-variate linear regression.

Reference material:
https://www.youtube.com/watch?v=VmbA0pi2cRQ
https://vitalflux.com/mean-square-error-r-squared-which-one-to-use/
https://vitalflux.com/interpreting-f-statistics-in-linear-regression-formula-examples/
"""


from dataclasses import InitVar, dataclass
import numpy as np
import util as u


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
                print(f'Epoch: {i}, current slope: {self.slope}, current intercept: {self.intercept}')
            self.__gradient_descent()
        print(f'The p value is {self.__calculate_p()}')
        return self

    def plot_line(self):
        return [self.slope*x + self.intercept for x in range(self.__predictor[0], self.__predictor[-1])]

    def __gradient_descent(self):
        """Attempts to minimize loss by way of gradient descent (loss function in this case is mean squared error(MSE)).

        Returns
        -------
            Parameters (slope and intercept) adjusted by negative gradient times the learning rate
        """
        slope_pd = 0
        intercept_pd = 0
        n = len(self.__predictor)

        for i in range(n):
            x = self.__predictor[i]
            y = self.__target[i]
            # calculate partial derivatives of MSE with respect to slope and intercept
            slope_pd += -(2/n) * x * (y - (self.slope*x + self.intercept))
            intercept_pd += -(2/n) * (y - (self.slope*x + self.intercept))

        self.slope -= slope_pd*self.__learn_rate
        self.intercept -= intercept_pd*self.__learn_rate

    def __calculate_p(self):
        se_mean = u.squared_error_total(self.__target)
        se_line = u.squared_error_regression(self.__predictor, self.__target, self.slope, self.intercept)
        # Degrees of freedom in the denominator,
        # number of observations minus 2 extra parameters in the model (slope and intercept)
        d_dof = len(self.__target) - 2
        var_explained = se_mean-se_line
        var_ratio = var_explained / (se_line/d_dof)
        # Note: degrees of freedom in the numerator are equal to 1 in the case of 2d linear model
        return u.f_test(var_ratio, 1, d_dof)
