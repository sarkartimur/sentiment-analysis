import logging
from math import floor
import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from src.linearregression.simple_lr import SimpleLinearRegression


PREDICTOR = np.array([5, 15, 25, 35, 45, 55])
SK_PREDICTOR = PREDICTOR.reshape(-1, 1)
TARGET = np.array([30, 21, 27, 39, 40, 37])
LEARN_RATE = 0.0001
EPOCHS = 25000

model = SimpleLinearRegression(PREDICTOR, TARGET, LEARN_RATE, EPOCHS)
# Note: sklearn uses the least squares method for optimization (solve Ax=b)
sk_model = LinearRegression(fit_intercept=True, copy_X=False, n_jobs=None)

# Compute slope using the formula
reference_slope = (np.corrcoef(PREDICTOR, TARGET)*TARGET.std()/PREDICTOR.std())[1][0]


def test_simple_linear_regression():
    model.fit()
    sk_model.fit(SK_PREDICTOR, TARGET)
    try:
        assert f'{reference_slope:.2f}' == f'{sk_model.coef_[0]:.2f}' == f'{model.slope:.2f}'
        assert floor(sk_model.intercept_) == floor(model.intercept)
    except AssertionError:
        fig, ax = plt.subplots()
        ax.scatter(PREDICTOR, TARGET, color='g')
        ax.plot(model.plot_line(), color='r')
        ax.plot(SK_PREDICTOR, sk_model.predict(SK_PREDICTOR), color='b')
        plt.savefig(f'simple_regression_{time.time()}_fail.png')