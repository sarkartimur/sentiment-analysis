"""
Contains a simple reference 'from scratch' implementation of singular linear regression.

Links to resources:
https://www.youtube.com/watch?v=VmbA0pi2cRQ
https://vitalflux.com/mean-square-error-r-squared-which-one-to-use/
"""


def gradient_descent(slope, intercept, inputs, outputs, learn_rate):
    """Attempts to minimize loss by way of gradient descent (loss function in this case is mean squared error(MSE)).

    Returns
    -------
        Parameters (slope and intercept) adjusted by negative gradient times the learning rate
    """
    slope_pd = 0
    intercept_pd = 0
    n = len(inputs)

    for i in range(n):
        # calculate partial derivatives of MSE with respect to slope and intercept
        slope_pd += -(2/n) * inputs[i] * (outputs[i] - (slope*inputs[i] + intercept))
        intercept_pd += -(2/n) * (outputs[i] - (slope*inputs[i] + intercept))

    slope -= slope_pd*learn_rate
    intercept -= intercept_pd*learn_rate

    return slope, intercept


def fit(epochs, m, b, x, y, lr):
    for i in range(epochs):
        if (i % 1000 == 0):
            print(f'Epoch: {i}, current slope: {m}, current intercept: {b}')
        m, b = gradient_descent(m, b, x, y, lr)

    return m, b        
