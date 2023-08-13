import statsmodels.api as sm
import time
from src.linearregression.gradient_descent import SquareErrorGradientDescentStrategy
from src.linearregression.linear_regression import GradLinearRegression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(777)


SHOW_PLOT = True
SAMPLE_SIZE = 50
# Note: the correllation between predictor and target in this case most likely will be weak,
# so the f-value will be small and the model will not be statistically significant
PREDICTOR = np.random.normal(0, 100, size=(SAMPLE_SIZE, 2))
TARGET = np.random.normal(0, 100, size=SAMPLE_SIZE)
LEARN_RATE = 0.01
EPOCHS = 10

strat = SquareErrorGradientDescentStrategy(LEARN_RATE)
model = GradLinearRegression(EPOCHS, strat)
# Note: sklearn uses the least squares method for optimization (solve Ax=b)
sk_model = LinearRegression(fit_intercept=False, copy_X=False, n_jobs=None)


def test_linear_regression():
    predictor, target = __standardize(PREDICTOR), __standardize(TARGET)
    model.fit(predictor, target)
    sk_model.fit(predictor, target)
    sm_model = sm.OLS(target, predictor)
    res = sm_model.fit()
    try:
        assert f'{sk_model.coef_[0]:.2f}' == f'{model.slope[0]:.2f}'
        assert f'{res.f_pvalue:.2f}' == f'{model.p_value:.2f}'
        if (SHOW_PLOT):
            __plot(predictor, target, model, sk_model)
            plt.show()
    except AssertionError as e:
        __plot(predictor, target, model, sk_model)
        plt.savefig(f'linear_regression_{time.time()}_fail.png')
        raise e


def __plot(predictor, target, model, sk_model):
    x, y = np.linspace(-2, 2, 10), np.linspace(-2, 2, 10)
    xs, ys = np.meshgrid(x, y)
    zs = xs*model.slope[0] + ys*model.slope[1]
    zs_sk = xs*sk_model.coef_[0] + ys*sk_model.coef_[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(azim=-60, elev=6)

    ax.scatter([i[0] for i in predictor], [i[1] for i in predictor], target, color='g')
    ax.plot_surface(xs, ys, zs, alpha=0.7, color='r')
    ax.plot_surface(xs, ys, zs_sk, alpha=0.5, color='b')


def __standardize(x: np.ndarray) -> np.ndarray:
    """
    Standardize data i.e. bring the mean to 0 (center data around 0),
    bring the standard deviation to 1.
    Note: standardization has the effect of replacing each value with it's z-score.

    Parameters
    ----------
    x
        Input array

    Returns
    -------
        Standardized array
    """
    mean = x.mean()
    std = x.std()
    return (x - mean)/std
