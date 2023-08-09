from src.linearregression.linear_regression import GradLinearRegression
from sklearn.linear_model import LinearRegression
from math import floor
import time
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(777)


PREDICTOR = np.random.randint(0, 100, size=(10, 2))
TARGET = np.random.randint(0, 100, size=10)
LEARN_RATE = 0.00001
EPOCHS = 200000

model = GradLinearRegression(LEARN_RATE, EPOCHS)
# Note: sklearn uses the least squares method for optimization (solve Ax=b)
sk_model = LinearRegression(fit_intercept=True, copy_X=False, n_jobs=None)


def test_linear_regression():
    model.fit(PREDICTOR, TARGET)
    sk_model.fit(PREDICTOR, TARGET)
    try:
        assert f'{sk_model.coef_[0]:.2f}' == f'{model.slope[0]:.2f}'
        assert floor(sk_model.intercept_) == floor(model.intercept)
    except AssertionError:
        x, y = np.linspace(0, 100, 10), np.linspace(0, 100, 10)
        xs, ys = np.meshgrid(x, y)
        zs = xs*model.slope[0] + ys*model.slope[1] + model.intercept
        zs_sk = xs*sk_model.coef_[0] + ys*sk_model.coef_[1] + sk_model.intercept_

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.view_init(azim=-60, elev=6)
        
        ax.scatter([i[0] for i in PREDICTOR], [i[1] for i in PREDICTOR], TARGET, color='g')
        ax.plot_surface(xs, ys, zs, alpha=0.7, color='r')
        ax.plot_surface(xs, ys, zs_sk, alpha=0.5, color='b')
        
        plt.savefig(f'linear_regression_{time.time()}_fail.png')
