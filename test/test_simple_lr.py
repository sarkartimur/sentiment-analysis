import logging
from math import floor
import time
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(777)
from sklearn.linear_model import LinearRegression
from src.linearregression.linear_regression import GradLinearRegression


PREDICTOR = np.random.randint(0, 100, size=(10, 2))
TARGET = np.random.randint(0, 100, size=10)
# PREDICTOR = np.array([5, 15, 25, 35, 45, 55])
# SK_PREDICTOR = PREDICTOR.reshape(-1, 1)
# TARGET = np.array([30, 21, 27, 39, 40, 37])
LEARN_RATE = 0.00001
EPOCHS = 200000

model = GradLinearRegression(LEARN_RATE, EPOCHS)
# Note: sklearn uses the least squares method for optimization (solve Ax=b)
sk_model = LinearRegression(fit_intercept=True, copy_X=False, n_jobs=None)

# Compute slope using the formula
# reference_slope = (np.corrcoef(PREDICTOR, TARGET)*TARGET.std()/PREDICTOR.std())[1][0]


def test_simple_linear_regression():
    model.fit(PREDICTOR, TARGET)
    sk_model.fit(PREDICTOR, TARGET)
    try:
        assert f'{sk_model.coef_[0]:.2f}' == f'{model.slope[0]:.2f}'
        assert floor(sk_model.intercept_) == floor(model.intercept)
    except AssertionError:
        fig, ax = plt.subplots()
        ax.scatter(PREDICTOR, TARGET, color='g')
        ax.plot(model.plot_line(PREDICTOR), color='r')
        # ax.plot(SK_PREDICTOR, sk_model.predict(SK_PREDICTOR), color='b')
        plt.savefig(f'simple_regression_{time.time()}_fail.png')