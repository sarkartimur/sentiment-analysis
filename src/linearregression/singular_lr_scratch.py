"""
Contains a simple reference 'from scratch' implementation of singular linear regression.

Links to resources:
https://www.youtube.com/watch?v=VmbA0pi2cRQ
https://vitalflux.com/mean-square-error-r-squared-which-one-to-use/
"""

from matplotlib import pyplot as plt
import numpy as np


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

    adj_slope = slope - slope_pd*learn_rate
    adj_intercept = intercept - intercept_pd*learn_rate

    return adj_slope, adj_intercept

x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([30, 21, 27, 39, 40, 37])
m = 0
b = 0
lr = 0.0001
epochs = 100000

for i in range(epochs):
    if (i % 100 == 0):
        print(f'Epoch: {i}, current slope: {m}, current intercept: {b}')
    m, b = gradient_descent(m, b, x, y, lr)

fig, ax = plt.subplots()
ax.scatter(x, y, color='g')
ax.plot([m*x + b for x in range(x[0], x[-1])])
plt.show()
