"""
Contains a simple reference 'from scratch' implementation of singular linear regression.

Reference material:
https://www.youtube.com/watch?v=VmbA0pi2cRQ
https://vitalflux.com/mean-square-error-r-squared-which-one-to-use/
https://vitalflux.com/interpreting-f-statistics-in-linear-regression-formula-examples/
"""


import numpy as np
import util as u


def gradient_descent(slope, intercept, predictor: np.ndarray, target: np.ndarray, learn_rate):
    """Attempts to minimize loss by way of gradient descent (loss function in this case is mean squared error(MSE)).

    Returns
    -------
        Parameters (slope and intercept) adjusted by negative gradient times the learning rate
    """
    slope_pd = 0
    intercept_pd = 0
    n = len(predictor)

    for i in range(n):
        # calculate partial derivatives of MSE with respect to slope and intercept
        slope_pd += -(2/n) * predictor[i] * (target[i] - (slope*predictor[i] + intercept))
        intercept_pd += -(2/n) * (target[i] - (slope*predictor[i] + intercept))

    slope -= slope_pd*learn_rate
    intercept -= intercept_pd*learn_rate

    return slope, intercept


def fit(epochs, m, b, x, y, lr):
    for i in range(epochs):
        if (i % 1000 == 0):
            print(f'Epoch: {i}, current slope: {m}, current intercept: {b}')
        m, b = gradient_descent(m, b, x, y, lr)

    # Compute the p-value
    se_mean = u.squared_error_mean(y)
    se_line = u.squared_error_line(x, y, m, b)
    # Degrees of freedom in the denominator,
    # number of observations minus 2 extra parameters in the model (slope and intercept)
    d_dof = len(y) - 2
    var_ratio = (se_mean-se_line) / (se_line/d_dof)
    # Note: degrees of freedom in the numerator are equal to 1 in the case of 2d linear model
    p_val = u.f_test(var_ratio, 1, d_dof)
    print(f'The p value is {p_val}')

    return m, b        
